
import os
import json
import re
from datetime import datetime, timezone

import httpx
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ================ CONFIGURATION ================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

PAN_PATTERN = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
DOB_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

REQUIRED_FIELDS = ["full_name", "pan_card", "dob", "skills"]


# ================ PAN VALIDATION ================

def validate_pan(pan):
    return (
        isinstance(pan, str)
        and bool(re.fullmatch(PAN_PATTERN, pan.strip()))
    )


# ================ DOB VALIDATION ================

def validate_dob(dob):
    if not isinstance(dob, str):
        return False

    if not re.fullmatch(DOB_PATTERN, dob):
        return False

    try:
        parsed_date = datetime.strptime(dob, "%Y-%m-%d").date()
        return parsed_date <= datetime.now().date()
    except ValueError:
        return False


# ================ JSON SCHEMA VALIDATION ================

def validate_extracted_data(data):
    if not isinstance(data, dict):
        return False, "Expected a JSON object"

    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"

    if not isinstance(data["full_name"], str):
        return False, "full_name must be a string"

    if not data["full_name"].strip():
        return False, "full_name cannot be empty"

    if not isinstance(data["pan_card"], str):
        return False, "pan_card must be a string"

    if not validate_pan(data["pan_card"]):
        return False, "Invalid or missing PAN format"

    if not isinstance(data["dob"], str):
        return False, "dob must be a string"

    if not validate_dob(data["dob"]):
        return False, "Invalid or missing DOB (YYYY-MM-DD)"

    if not isinstance(data["skills"], list):
        return False, "skills must be a list"

    if not all(isinstance(skill, str) for skill in data["skills"]):
        return False, "Every skill must be a string"

    return True, "Validation successful"


# ================ DEFENSIVE JSON PARSING ================

def parse_json_response(response_text):
    if not isinstance(response_text, str):
        return None, "PARSING_ERROR: Response is not text"

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return None, "PARSING_ERROR: Invalid JSON response"

    valid, message = validate_extracted_data(data)

    if not valid:
        return None, f"VALIDATION_FAILED: {message}"

    return data, "SUCCESS"


# ================ XML-STYLE PROMPT ================

def build_extraction_prompt(resume_text, repair_error=None):
    repair_instruction = ""

    if repair_error:
        repair_instruction = (
            f"\nPrevious response failed validation: {repair_error}\n"
            "Correct the response and return valid JSON only.\n"
        )

    return f"""
Extract candidate information from the resume below.

Return exactly one JSON object containing:
- full_name: string
- pan_card: string
- dob: string in YYYY-MM-DD format
- skills: list of strings

For missing name, PAN, or DOB, use an empty string.
For missing skills, use an empty list.
Never invent personal information.
Return JSON only, without Markdown code fences.

{repair_instruction}
<data>
{resume_text}
</data>
""".strip()


# ================ OPENROUTER CLIENT ================

def create_ai_client():
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if (
        not api_key
        or api_key == "your_openrouter_api_key_here"
    ):
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    http_client = httpx.Client(
        trust_env=False,
        timeout=45.0
    )

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        http_client=http_client,
        max_retries=1
    )


# ================ AI RESUME EXTRACTION ================

def extract_resume_with_ai(resume_text):
    client = None

    try:
        client = create_ai_client()
        repair_error = None

        # Initial attempt plus one repair attempt.
        for attempt in range(2):
            prompt = build_extraction_prompt(
                resume_text,
                repair_error
            )

            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract resume information accurately. "
                            "Return JSON with full_name, pan_card, "
                            "dob, and skills. Never invent information."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result, status = parse_json_response(content)

            if result is not None:
                print("AI extraction and validation: SUCCESS")
                return result, "SUCCESS"

            repair_error = status
            print(f"AI attempt {attempt + 1}: {status}")

        return None, f"VALIDATION_FAILED: {repair_error}"

    except Exception as error:
        print("AI request failed:", type(error).__name__)
        return None, "AI_REQUEST_FAILED"

    finally:
        if client is not None:
            client.close()


# ================ SHEETDB STORAGE ================

def store_in_sheetdb(data):
    sheetdb_url = os.getenv("SHEETDB_API_URL", "").strip()

    if not sheetdb_url:
        return False, "SHEETDB_NOT_CONFIGURED"

    row = {
        "full_name": data["full_name"],
        "pan_card": data["pan_card"],
        "dob": data["dob"],
        "skills": ", ".join(data["skills"]),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        # Direct connection, bypassing environment proxies.
        with requests.Session() as session:
            session.trust_env = False

            response = session.post(
                sheetdb_url,
                json={"data": [row]},
                timeout=20
            )

        if response.ok:
            print("SheetDB write: SUCCESS")
            print("SheetDB response:", response.text[:500])
            return True, "SHEETDB_SUCCESS"

        print("SheetDB HTTP status:", response.status_code)
        print("SheetDB error:", response.text[:1000])
        return False, "SHEETDB_FAILED"

    except requests.RequestException as error:
        print("SheetDB request failed:", type(error).__name__)
        return False, "SHEETDB_CONNECTION_FAILED"


# ================ FIREBASE CLOUD FIRESTORE ================

def write_firebase_audit(event, status, details=None):
    credential_path = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", ""
    ).strip()

    project_id = os.getenv(
        "FIREBASE_PROJECT_ID", ""
    ).strip()

    if not credential_path or not os.path.isfile(credential_path):
        return False, "FIREBASE_CREDENTIALS_NOT_CONFIGURED"

    if not project_id:
        return False, "FIREBASE_PROJECT_ID_NOT_CONFIGURED"

    google_session = None

    try:
        import firebase_admin
        import google.auth.transport.requests as google_requests
        from firebase_admin import credentials, firestore

        # Use a session that bypasses environment proxy settings.
        google_session = requests.Session()
        google_session.trust_env = False

        auth_request = google_requests.Request(
            session=google_session
        )

        try:
            app = firebase_admin.get_app()
        except ValueError:
            certificate = credentials.Certificate(credential_path)

            app = firebase_admin.initialize_app(
                certificate,
                {"projectId": project_id}
            )

        # Refresh the service-account credentials directly.
        google_credential = app.credential.get_credential()
        google_credential.refresh(auth_request)

        # Firestore's gRPC transport can still have separate
        # proxy or network requirements.
        client = firestore.client(app=app)

        audit_record = {
            "event": event,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        client.collection("day15_audit_logs").add(audit_record)

        print("Firebase Firestore audit log: SUCCESS")
        return True, "FIREBASE_AUDIT_SUCCESS"

    except Exception as error:
        print("Firebase audit failed:", type(error).__name__)
        print("Error details:", str(error)[:1000])
        return False, "FIREBASE_AUDIT_FAILED"

    finally:
        if google_session is not None:
            google_session.close()


# ================ END-TO-END PIPELINE ================

def process_resume(resume_text):
    print("\n--- Starting Resume Pipeline ---")

    extracted_data, ai_status = extract_resume_with_ai(resume_text)

    if extracted_data is None:
        _, firebase_status = write_firebase_audit(
            event="resume_processing",
            status="FAILED",
            details={"stage": ai_status}
        )

        result = {
            "status": "FAILED",
            "reason": ai_status,
            "sheetdb": "NOT_ATTEMPTED",
            "firebase": firebase_status
        }

        print("\n--- Pipeline Result ---")
        print(json.dumps(result, indent=4))
        return result

    print("\n--- Validated Extracted Data ---")
    print(json.dumps(extracted_data, indent=4))

    sheetdb_ok, sheetdb_status = store_in_sheetdb(extracted_data)

    _, firebase_status = write_firebase_audit(
        event="resume_processing",
        status="SUCCESS" if sheetdb_ok else "PARTIAL_SUCCESS",
        details={
            "validation": "SUCCESS",
            "sheetdb_status": sheetdb_status
        }
    )

    overall_success = (
        sheetdb_ok
        and firebase_status == "FIREBASE_AUDIT_SUCCESS"
    )

    result = {
        "status": "SUCCESS" if overall_success else "PARTIAL_SUCCESS",
        "validation": "SUCCESS",
        "sheetdb": sheetdb_status,
        "firebase": firebase_status
    }

    print("\n--- Pipeline Result ---")
    print(json.dumps(result, indent=4))

    return result


# ================ RUN PROGRAM ================

if __name__ == "__main__":
    sample_resume = """
    Name: Ravi Kumar
    PAN: ABCDE1234F
    Date of Birth: 1998-05-12
    Skills: Python, Java
    """

    process_resume(sample_resume)
