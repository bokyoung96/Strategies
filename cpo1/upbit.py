import re
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

from cpo1.base import NoticeCrawler
from cpo1.market import Market
from cpo1.models import ListingDetail, NoticeSummary
from cpo1.session import BrowserSession

BASE = "https://upbit.com"
LIST_PAGE = f"{BASE}/service_center/notice"


class UpbitNoticeCrawler(NoticeCrawler):
    _RE_NEW_LISTING = re.compile(r"신규\s*거래\s*지원", re.IGNORECASE)
    _RE_TICKER = re.compile(
        r"[가-힣A-Za-z\s\-/]+?\(([A-Z0-9\-]{2,15})\)\s*신규\s*거래\s*지원")
    _RE_DATETIME = r"(\d{4}[.\-]\d{2}[.\-]\d{2})[^0-9]{0,6}(\d{2}:\d{2})"
    _RE_DEPOSIT = re.compile(r"(입금|디파짓).{0,20}" + _RE_DATETIME)
    _RE_TRADE = re.compile(r"(거래\s*지원|트레이딩\s*오픈|상장).{0,20}" + _RE_DATETIME)
    _RE_WITHDRAW = re.compile(r"(출금|위드드로).{0,20}" + _RE_DATETIME)

    def __init__(self, session: BrowserSession, market_mode: Market = None):
        self.session = session
        self.market_mode = market_mode or Market.KRW

    @staticmethod
    def _extract_id_from_url(url: str) -> Optional[int]:
        try:
            q = parse_qs(urlparse(url).query)
            v = q.get("id", [None])[0]
            return int(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _extract_ticker_from_title(title: str) -> Optional[str]:
        m = UpbitNoticeCrawler._RE_TICKER.search(title)
        return m.group(1) if m else None

    def _extract_markets(self, text: str) -> List[str]:
        return Market.from_text(text, self.market_mode)

    def iter_new_listings(self, limit: int = 50) -> List[NoticeSummary]:
        page = self.session.new_page()
        page.goto(LIST_PAGE, timeout=60_000)
        page.wait_for_load_state("networkidle", timeout=45_000)

        anchors: List[Dict[str, str]] = page.evaluate("""
        () => Array.from(document.querySelectorAll("a"))
            .map(a => ({ href: a.getAttribute("href") || "", title: (a.innerText || "").trim() }))
            .filter(x => x.href.includes('/service_center/notice') && x.title.length > 0);
        """)
        seen = set()
        results: List[NoticeSummary] = []

        for a in anchors:
            title = a["title"]
            href = a["href"]
            if not self._RE_NEW_LISTING.search(title):
                continue

            url = urljoin(BASE, href)
            if url in seen:
                continue
            seen.add(url)

            ticker = self._extract_ticker_from_title(title)
            markets = self._extract_markets(title)

            results.append(
                NoticeSummary(
                    id=self._extract_id_from_url(url),
                    title=title,
                    url=url,
                    ticker=ticker,
                    markets=markets,
                )
            )
            if len(results) >= limit:
                break

        page.context.close()
        return results

    def fetch_detail(self, url: str, hint: Optional[NoticeSummary] = None) -> ListingDetail:
        page = self.session.new_page()
        page.goto(url, timeout=60_000)
        page.wait_for_load_state("networkidle", timeout=45_000)

        content_text: str = page.evaluate("""
        () => {
            const candidates = [
                'main', 'article', '.article', '.content', '.board-view', '.sc-bdvvtL', '.sc-',
                '#__next', 'body'
            ];
            for (const sel of candidates) {
                const el = document.querySelector(sel);
                if (el && (el.innerText || '').trim().length > 200) {
                    return el.innerText.trim();
                }
            }
            return document.body.innerText.trim();
        }
        """)

        title_text: str = page.title() or (hint.title if hint else "")
        combined = f"{title_text}\n\n{content_text}"

        def grab_first(re_pat: re.Pattern) -> Optional[str]:
            m = re_pat.search(combined)
            if not m:
                return None
            date_part = m.group(
                2) if m.lastindex and m.lastindex >= 2 else None
            time_part = m.group(
                3) if m.lastindex and m.lastindex >= 3 else None
            if date_part and time_part:
                return f"{date_part} {time_part}"
            return m.group(0)

        trade_open = grab_first(self._RE_TRADE)
        deposit_open = grab_first(self._RE_DEPOSIT)
        withdraw_open = grab_first(self._RE_WITHDRAW)

        ticker = (hint.ticker if hint else None) or self._extract_ticker_from_title(
            title_text)
        markets = (hint.markets if hint else None) or self._extract_markets(
            title_text)
        if not markets:
            markets = self._extract_markets(content_text)

        detail = ListingDetail(
            id=self._extract_id_from_url(url),
            title=title_text.strip(),
            url=url,
            ticker=ticker,
            markets=markets,
            trade_open_text=trade_open,
            deposit_open_text=deposit_open,
            withdraw_open_text=withdraw_open,
            content_text=content_text,
        )
        page.context.close()
        return detail
