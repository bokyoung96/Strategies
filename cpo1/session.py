from typing import Optional

from playwright.sync_api import Browser, Page, sync_playwright


class BrowserSession:
    def __init__(self, headless: bool = True, user_agent: Optional[str] = None):
        self.headless = headless
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115 Safari/537.36"
        )
        self._p = None
        self._browser: Optional[Browser] = None

    def __enter__(self):
        self._p = sync_playwright().start()
        self._browser = self._p.chromium.launch(
            headless=self.headless, args=["--no-sandbox"]
        )
        return self

    def new_page(self) -> Page:
        assert self._browser is not None
        context = self._browser.new_context(user_agent=self.user_agent)
        return context.new_page()

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._p:
                self._p.stop()
