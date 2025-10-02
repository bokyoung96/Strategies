import json

import requests

BASE = "https://api-manager.upbit.com/api/v1/announcements"

url = f"{BASE}?os=web&page=1&per_page=3&category=trade"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
data = resp.json()

print(json.dumps(data, ensure_ascii=False, indent=2))
