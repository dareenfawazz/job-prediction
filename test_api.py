# test_api.py (Final Testing Script - No changes needed for SHAP)

import requests
import json
import time

# The URL of your running Flask server's prediction endpoint
API_URL = "http://127.0.0.1:5000/predict"

# --- Test Case: Real/Legitimate Job ---
test_data = {
    "title": ["Senior Developer"],
    "location": ["CA, USA"],
    "department": ["IT"],
    "company_profile": ["We are a stable tech firm."],
    "description": ["Must have 5+ years experience in Python and cloud."],
    "requirements": ["BS/MS in CS"],
    "benefits": ["Full benefits package."],
    "employment_type": ["Full-time"],
    "required_experience": ["Mid-Senior level"],
    "required_education": ["Bachelor's Degree"],
    "industry": ["Information Technology and Services"],
    "function": ["Information Technology"],
    "telecommuting": [0],
    "has_company_logo": [1],
    "has_questions": [1]
}

# --- Send the Request ---
print("Sending test request to API...")
time.sleep(1) 

try:
    # Send the request with the JSON payload
    response = requests.post(API_URL, json=test_data)
    
    # Check for success (HTTP 200)
    if response.status_code == 200:
        print("\n✅ API Response Successful (Status 200)")
        print("-" * 35)
        
        # Pretty-print the JSON prediction
        prediction_result = response.json()
        print(json.dumps(prediction_result, indent=4))
        
    else:
        print(f"\n❌ API Error: Status Code {response.status_code}")
        print("Response Text (Error from Server):", response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR: The Flask server is not running or the URL is wrong.")
    print("Please ensure 'python app.py' is running in another terminal window.")