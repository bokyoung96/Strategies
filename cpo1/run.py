import json
import os
import sys

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cpo1.main import get_upbit_new_listings

if __name__ == "__main__":
    data = get_upbit_new_listings(max_items=8, headless=True)
    print(json.dumps(data["summaries"], ensure_ascii=False, indent=2))
