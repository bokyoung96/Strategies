import requests


class ApiSession:
    def __init__(self, user_agent: str | None = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or "Mozilla/5.0 (compatible; UpbitCrawler/1.0)"
        })

    def get(self, url: str, **kwargs):
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
