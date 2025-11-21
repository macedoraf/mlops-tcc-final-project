"""
Python test script for the Sentiment Analysis API.
Tests all endpoints including translation, feedback, and metrics.
"""
import requests
import json
from typing import Dict, Any

API_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(success: bool, message: str, data: Any = None):
    """Print test result."""
    status = "✓" if success else "✗"
    print(f"{status} {message}")
    if data:
        print(f"   Response: {json.dumps(data, indent=2)}")


def test_health():
    """Test health endpoint."""
    print_section("1. Testing Health Endpoint")
    try:
        response = requests.get(f"{API_URL}/health")
        success = response.status_code == 200
        print_result(success, f"Health check (HTTP {response.status_code})", response.json())
        return success
    except Exception as e:
        print_result(False, f"Health check failed: {e}")
        return False


def test_predict_english():
    """Test prediction with English text."""
    print_section("2. Testing Prediction with English Text")
    try:
        payload = {
            "text": "This product is absolutely amazing! I love it!",
            "language": "en"
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        success = response.status_code == 200
        data = response.json()
        
        print_result(success, f"English prediction (HTTP {response.status_code})", data)
        
        if success:
            return data.get("prediction_id")
        return None
    except Exception as e:
        print_result(False, f"English prediction failed: {e}")
        return None


def test_predict_portuguese():
    """Test prediction with Portuguese text (translation)."""
    print_section("3. Testing Prediction with Portuguese Text (Translation)")
    try:
        payload = {
            "text": "Este produto é excelente! Estou muito satisfeito com a compra.",
            "language": "pt"
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        success = response.status_code == 200
        data = response.json()
        
        has_translation = data.get("translated_text") is not None if success else False
        print_result(
            success and has_translation,
            f"Portuguese prediction with translation (HTTP {response.status_code})",
            data
        )
        
        if success:
            return data.get("prediction_id")
        return None
    except Exception as e:
        print_result(False, f"Portuguese prediction failed: {e}")
        return None


def test_feedback(prediction_id: str):
    """Test feedback endpoint."""
    print_section("4. Testing Feedback Endpoint")
    if not prediction_id:
        print_result(False, "Skipping feedback test (no prediction ID)")
        return False
    
    try:
        payload = {
            "prediction_id": prediction_id,
            "correct_sentiment": 1
        }
        response = requests.post(f"{API_URL}/feedback", json=payload)
        success = response.status_code == 200
        data = response.json()
        
        print_result(success, f"Feedback submission (HTTP {response.status_code})", data)
        return success
    except Exception as e:
        print_result(False, f"Feedback submission failed: {e}")
        return False


def test_metrics():
    """Test real-time metrics endpoint."""
    print_section("5. Testing Real-time Metrics Endpoint")
    try:
        response = requests.get(f"{API_URL}/metrics/realtime")
        success = response.status_code == 200
        data = response.json()
        
        print_result(success, f"Metrics retrieval (HTTP {response.status_code})", data)
        return success
    except Exception as e:
        print_result(False, f"Metrics retrieval failed: {e}")
        return False


def test_negative_sentiment():
    """Test negative sentiment detection."""
    print_section("6. Testing Negative Sentiment (Portuguese)")
    try:
        payload = {
            "text": "Produto horrível! Não recomendo. Péssima qualidade.",
            "language": "pt"
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        success = response.status_code == 200
        data = response.json()
        
        is_negative = data.get("sentiment") == "NEGATIVE" if success else False
        print_result(
            success and is_negative,
            f"Negative sentiment detection (HTTP {response.status_code})",
            data
        )
        return success and is_negative
    except Exception as e:
        print_result(False, f"Negative sentiment detection failed: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid input."""
    print_section("7. Testing Error Handling")
    try:
        # Test with empty text
        payload = {
            "text": "",
            "language": "en"
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        handles_empty = response.status_code == 422  # Validation error expected
        
        print_result(
            handles_empty,
            f"Empty text validation (HTTP {response.status_code})",
            response.json() if handles_empty else None
        )
        
        # Test with invalid feedback
        payload = {
            "prediction_id": "invalid-uuid-12345",
            "correct_sentiment": 1
        }
        response = requests.post(f"{API_URL}/feedback", json=payload)
        handles_invalid = response.status_code == 404  # Not found expected
        
        print_result(
            handles_invalid,
            f"Invalid prediction ID handling (HTTP {response.status_code})",
            response.json() if handles_invalid else None
        )
        
        return handles_empty and handles_invalid
    except Exception as e:
        print_result(False, f"Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  FastAPI Sentiment Analysis API Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    
    prediction_id_en = test_predict_english()
    results.append(("English Prediction", prediction_id_en is not None))
    
    prediction_id_pt = test_predict_portuguese()
    results.append(("Portuguese Prediction", prediction_id_pt is not None))
    
    results.append(("Feedback Submission", test_feedback(prediction_id_en)))
    results.append(("Metrics Retrieval", test_metrics()))
    results.append(("Negative Sentiment", test_negative_sentiment()))
    results.append(("Error Handling", test_error_handling()))
    
    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\nTo view production logs:")
    print("  docker exec -it <container_name> cat /app/logs/production_logs.jsonl")
    print()


if __name__ == "__main__":
    main()
