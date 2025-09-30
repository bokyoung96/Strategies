import json
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE = "https://upbit.com"
LIST_PAGE = f"{BASE}/service_center/notice"

def fetch_notices(headless=True):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/115 Safari/537.36"
        )
        page = context.new_page()
        page.goto(LIST_PAGE, timeout=60000)
        page.wait_for_load_state("networkidle", timeout=45000)

        data = page.evaluate("""
        () => {
            return Array.from(document.querySelectorAll("a"))
                .filter(a => (a.getAttribute("href") || "").includes('/service_center/notice'))
                .map(a => ({
                    title: a.innerText.trim(),
                    href: a.getAttribute("href")
                }))
                .filter(x => x.title.length > 0);
        }
        """)

        results = [{"title": r["title"], "url": urljoin(BASE, r["href"])} for r in data]
        browser.close()
        return results

def main():
    notices = fetch_notices(headless=True)
    print("found:", len(notices))
    print(json.dumps(notices[:10], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
