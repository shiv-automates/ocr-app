import requests
import json

url = "https://api.github.com/user/repos"
headers = {
    "Authorization": "token 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",
    "Accept": "application/vnd.github.v3+json"
}
payload = {
    "name": "ocr-app",
    "private": True
}

try:
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 201:
        print("SUCCESS: Repository 'ocr-app' created successfully on GitHub!")
    elif resp.status_code == 422:
        # 422 typically means the repository already exists for this user
        print("SUCCESS: Repository 'ocr-app' already exists, proceeding...")
    else:
        print(f"ERROR: Failed to create repository. Status code: {resp.status_code}")
        print(resp.text)
except Exception as e:
    print(f"ERROR: Exception occurred: {e}")
