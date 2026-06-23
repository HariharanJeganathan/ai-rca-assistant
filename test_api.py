"""
test_api.py — API Endpoint Tests
==================================
Tests verify that the API endpoints work correctly.

WHY TESTS MATTER FOR YOUR PORTFOLIO:
  - Shows you write production-quality code
  - GitHub Actions runs these automatically on every push
  - Interviewers check if you have tests — many candidates don't
  - Shows you know pytest, the industry-standard Python test framework

HOW TO RUN:
  cd backend
  pytest tests/ -v

WHAT THE -v FLAG DOES:
  Without -v:  just shows PASSED/FAILED count
  With -v:     shows each test name and its result (more useful)
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import sys
import os

# Add backend to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables BEFORE importing app
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("CHROMA_PERSIST_PATH", "/tmp/chroma_test")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")

from main import app

# TestClient = FastAPI's built-in test helper
# It lets us make fake HTTP requests without running a real server
client = TestClient(app, raise_server_exceptions=False)


# ============================================================
# Test 1: Root Endpoint
# ============================================================
def test_root_endpoint():
    """
    Test that the root URL returns 200 OK.
    This is the most basic test — if this fails, nothing works.
    """
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    print(f"✅ Root endpoint works: {data['message']}")


# ============================================================
# Test 2: Health Check
# ============================================================
def test_health_check():
    """
    Test the /health endpoint.
    In production, Render/Railway call this every 30 seconds
    to check if the app is alive.
    """
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data
    print(f"✅ Health check works: status={data['status']}")


# ============================================================
# Test 3: Swagger Docs Are Available
# ============================================================
def test_docs_endpoint():
    """
    Test that the auto-generated Swagger UI docs load.
    This is what interviewers see when they open your API.
    """
    response = client.get("/docs")
    assert response.status_code == 200
    print("✅ Swagger docs are accessible at /docs")


# ============================================================
# Test 4: Submit Incident for RCA
# ============================================================
def test_analyze_incident():
    """
    Test the main RCA analysis endpoint.
    Submits a sample incident and checks the response structure.
    """
    sample_incident = {
        "incident": {
            "title": "Payment service returning 502 errors",
            "description": (
                "Users are unable to complete checkout. "
                "Payment service is returning 502 Bad Gateway errors. "
                "Error rate spiked to 95% at 10:00 AM. "
                "This happened 5 minutes after deploying v2.3.1."
            ),
            "severity": "P1",
            "affected_systems": ["payment-service", "checkout-api"],
            "incident_timeline": (
                "09:55 - Deployment of v2.3.1 pushed to production\n"
                "10:00 - Alerts fired for payment-service errors\n"
                "10:05 - On-call engineer paged\n"
                "10:20 - Rollback initiated\n"
                "10:35 - Service restored"
            ),
            "additional_context": "Deployment included a database schema change"
        }
    }

    response = client.post("/api/v1/rca/analyze", json=sample_incident)

    assert response.status_code == 200
    data = response.json()

    # Check response has required fields
    assert data["success"] is True
    assert "incident_id" in data
    assert "status" in data
    assert data["incident_id"].startswith("INC-")

    print(f"✅ RCA analyze works: incident_id={data['incident_id']}")


# ============================================================
# Test 5: Validation — Missing Required Fields
# ============================================================
def test_analyze_incident_missing_title():
    """
    Test that the API rejects requests with missing required fields.
    Pydantic handles this automatically — we just verify it works.
    """
    bad_request = {
        "incident": {
            # Missing "title" which is required
            "description": "Something went wrong"
        }
    }

    response = client.post("/api/v1/rca/analyze", json=bad_request)

    # Should return 422 Unprocessable Entity (FastAPI's validation error)
    assert response.status_code == 422
    print("✅ Validation works: correctly rejected missing title")


# ============================================================
# Test 6: Validation — Title Too Short
# ============================================================
def test_analyze_incident_title_too_short():
    """
    Test that the API rejects titles shorter than 5 characters.
    (We set min_length=5 in schemas.py)
    """
    bad_request = {
        "incident": {
            "title": "Hi",    # Only 2 chars, minimum is 5
            "description": "Something went wrong with the service today and users are affected"
        }
    }

    response = client.post("/api/v1/rca/analyze", json=bad_request)
    assert response.status_code == 422
    print("✅ Validation works: correctly rejected short title")


# ============================================================
# Test 7: List Reports — Empty at Start
# ============================================================
def test_list_reports_empty():
    """
    Test that listing reports works even when there are none.
    Should return an empty list, not an error.
    """
    response = client.get("/api/v1/rca/reports")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "reports" in data
    print(f"✅ List reports works: found {len(data['reports'])} reports")


# ============================================================
# Test 8: Get Report That Doesn't Exist
# ============================================================
def test_get_nonexistent_report():
    """
    Test that fetching a non-existent report returns 404.
    Good APIs return meaningful error codes, not crashes.
    """
    response = client.get("/api/v1/rca/reports/INC-DOESNOTEXIST-999")
    assert response.status_code == 404
    print("✅ 404 handling works: correctly returned 404 for missing report")


# ============================================================
# Test 9: Knowledge Base Search
# ============================================================
def test_search_knowledge_base():
    """
    Test the knowledge base search endpoint.
    """
    response = client.get("/api/v1/incidents/search?query=database+connection+failure")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data
    print("✅ Knowledge base search works")


# ============================================================
# Test 10: File Upload — Wrong File Type
# ============================================================
def test_upload_wrong_file_type():
    """
    Test that uploading a .docx file (not supported) returns 400.
    """
    import io
    fake_file = io.BytesIO(b"fake content")

    response = client.post(
        "/api/v1/rca/upload",
        files={"file": ("incident.docx", fake_file, "application/vnd.openxmlformats")}
    )

    assert response.status_code == 400
    print("✅ File validation works: correctly rejected unsupported file type")


# ============================================================
# Run tests directly (python test_api.py)
# ============================================================
if __name__ == "__main__":
    print("\n🧪 Running API Tests...\n")
    test_root_endpoint()
    test_health_check()
    test_docs_endpoint()
    test_analyze_incident()
    test_analyze_incident_missing_title()
    test_analyze_incident_title_too_short()
    test_list_reports_empty()
    test_get_nonexistent_report()
    test_search_knowledge_base()
    test_upload_wrong_file_type()
    print("\n✅ All tests passed!\n")
