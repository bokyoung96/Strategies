import re
import time
from datetime import datetime
from functools import cached_property
from typing import Generator, List, Optional, Tuple

import pandas as pd
from cpo1.base import NoticeCrawler
from cpo1.market import Market
from cpo1.models import ListingDetail, ListingEntry, NoticeSummary
from cpo1.session import ApiSession
from tqdm import tqdm

SEARCH_API = "https://api-manager.upbit.com/api/v1/announcements/search"
DETAIL_API = "https://api-manager.upbit.com/api/v1/announcements/{id}"
WEB_BASE = "https://upbit.com/service_center/notice?id="


class UpbitNoticeCrawler(NoticeCrawler):
    _RE_PAIR = re.compile(
        r"\((?P<ticker>[A-Z0-9\-]{2,15})\)[^()]*\((?P<markets>[^)]+)\s*마켓\)"
    )
    _RE_TICKERS = re.compile(r"\(([A-Z0-9\-]{2,15})\)")

    # NOTE: Listing detail patterns for trade open time
    _RE_TRADE_ORIGINAL = re.compile(
        r"(?:거래지원\s*개시\s*시점)[^\d]*(\d{4}[.\-]\d{2}[.\-]\d{2}[^\d]{0,3}\d{1,2}:\d{2})"
    )
    _RE_TRADE_UPDATED = re.compile(
        r"(?:변경된|연기된)\s*거래지원\s*개시\s*시점[^\d]*(\d{4}[.\-]\d{2}[.\-]\d{2}[^\d]{0,3}\d{1,2}:\d{2})"
    )
    _RE_TRADE_PREV = re.compile(
        r"기존\s*거래지원\s*개시\s*시점[^\d]*(\d{4}[.\-]\d{2}[.\-]\d{2}[^\d]{0,3}\d{1,2}:\d{2})"
    )
    _RE_TRADE_KR = re.compile(r"(\d{1,2})월\s*(\d{1,2})일\s*(\d{1,2})시")

    def __init__(
        self,
        session: ApiSession,
        market_mode: Market = Market.KRW,
        limit: int = 50,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ):
        self.session = session
        self.market_mode = market_mode
        self.limit = limit
        self.start_page = start_page
        self.end_page = end_page

    def _iter_notices(self, page_size: int = 20) -> Generator[dict, None, None]:
        page = self.start_page
        count = 0

        while True:
            if self.end_page is not None and page > self.end_page:
                break
            if count >= self.limit:
                break

            data = self.session.get(SEARCH_API, params={
                "os": "web",
                "page": page,
                "per_page": page_size,
                "category": "all",
                "search": "신규 거래지원",
            })
            notices = data["data"]["list"]
            if not notices:
                break

            for n in notices:
                yield n
                count += 1
                if count >= self.limit:
                    return
            page += 1

    def _extract_pairs(self, title: str) -> List[ListingEntry]:
        pairs: List[ListingEntry] = []
        for m in self._RE_PAIR.finditer(title):
            ticker = m.group("ticker")
            markets = [s.strip().upper()
                       for s in m.group("markets").split(",")]
            pairs.append(ListingEntry(ticker=ticker, markets=markets))

        if not pairs:
            tickers = self._RE_TICKERS.findall(title)
            markets = Market.from_text(title, self.market_mode)
            pairs = [ListingEntry(t, markets) for t in tickers]
        return pairs

    def _get_year(self, month: str, day: str, hour: str, listed_at: Optional[str]) -> str:
        try:
            base_year = datetime.fromisoformat(
                listed_at).year if listed_at else datetime.now().year
            return f"{base_year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:00"
        except Exception:
            return f"{month}월 {day}일 {hour}시"

    def _extract_trade_times(self, text: str, listed_at: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        trade_open_original, trade_open_updated = None, None

        patterns = [
            ("updated", self._RE_TRADE_UPDATED, None),
            ("original", self._RE_TRADE_PREV, None),
            ("original", self._RE_TRADE_ORIGINAL, None),
            ("original", self._RE_TRADE_KR,
             lambda m: self._get_year(m.group(1), m.group(2), m.group(3), listed_at)),
        ]

        for kind, regex, transform in patterns:
            m = regex.search(text)
            if m:
                value = transform(m) if transform else m.group(1)
                if kind == "updated":
                    trade_open_updated = value
                elif not trade_open_original:
                    trade_open_original = value
        return trade_open_original, trade_open_updated

    @cached_property
    def summaries(self) -> List[NoticeSummary]:
        results: List[NoticeSummary] = []
        for n in self._iter_notices():
            results.append(
                NoticeSummary(
                    id=n["id"],
                    title=n["title"],
                    url=f"{WEB_BASE}{n['id']}",
                    listings=self._extract_pairs(n["title"]),
                )
            )
        return results

    @cached_property
    def details(self) -> List[ListingDetail]:
        results = []
        for summary in tqdm(self.summaries, desc="Fetching details", unit="notice"):
            results.append(self.fetch_detail(summary.id, summary))
        return results

    def fetch_detail(self, notice_id: int, hint: Optional[NoticeSummary] = None) -> ListingDetail:
        # NOTE: To prevent from rate-limit overflow
        url = DETAIL_API.format(id=notice_id)
        time.sleep(1.0)

        data = self.session.get(url)["data"]
        content_html = data.get("body", "")
        content_text = re.sub(r"<[^>]+>", "", content_html)

        trade_open_original, trade_open_updated = self._extract_trade_times(
            content_text, data.get("listed_at")
        )

        return ListingDetail(
            id=data["id"],
            title=data["title"],
            url=hint.url if hint else f"{WEB_BASE}{data['id']}",
            listings=hint.listings if hint else [],
            trade_open_original=trade_open_original,
            trade_open_updated=trade_open_updated,
            content_text=content_text,
            category=data.get("category"),
            need_new_badge=data.get("need_new_badge"),
            need_update_badge=data.get("need_update_badge"),
            listed_at=data.get("listed_at"),
            first_listed_at=data.get("first_listed_at"),
        )

    def crawled(self) -> pd.DataFrame:
        rows = []
        for d in self.details:
            for entry in d.listings:
                rows.append({
                    "ticker": entry.ticker,
                    "markets": ",".join(entry.markets),
                    "trade_open_original": d.trade_open_original,
                    "trade_open_updated": d.trade_open_updated,
                    "trade_open": d.trade_open_updated or d.trade_open_original,
                })
        return pd.DataFrame(rows)
