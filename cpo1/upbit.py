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

# =============================================================================
# 📘 UpbitNoticeCrawler — Listing Table Parsing Specification
# =============================================================================
#
# Overview
# --------
# The UpbitNoticeCrawler is designed to extract new listing announcements
# ("신규 거래지원 안내") from Upbit's official notice API.
# While most listings include a single asset per notice, certain announcements
# contain multiple listings with **distinct markets or trade opening times**.
# To accurately capture those cases, this crawler analyzes both:
#
#   (1) The notice title  → used as a fallback (simple cases)
#   (2) The notice body   → parsed for detailed per-asset info (table-based)
#
# -----------------------------------------------------------------------------
# 1. Base Data Models
# -----------------------------------------------------------------------------
#
# Each listing is represented as a `ListingEntry` object:
#
#   ListingEntry(
#       ticker: str,          # Asset symbol, e.g. "AKT"
#       markets: List[str],   # Supported markets, e.g. ["KRW", "BTC", "USDT"]
#       trade_open: Optional[str]  # Parsed trade opening datetime (KST)
#   )
#
# The crawler aggregates these into a higher-level `ListingDetail`:
#
#   ListingDetail(
#       id: int,
#       title: str,
#       listings: List[ListingEntry],
#       trade_open_original: Optional[str],
#       trade_open_updated: Optional[str],
#       ...
#   )
#
# -----------------------------------------------------------------------------
# 2. Table-Based Parsing Logic
# -----------------------------------------------------------------------------
#
# Some Upbit announcements include multiple listings under a single notice.
# In such cases, the HTML body contains a <table> where **each <tr> row
# represents one digital asset** (ticker) with its corresponding markets
# and an independent trade opening time.
#
# Example Table:
#   <table>
#     <thead>
#       <tr>
#         <th>디지털 자산</th><th>마켓</th><th>네트워크</th>
#         <th>입출금 개시 시점</th><th>거래지원 개시 시점</th>
#       </tr>
#     </thead>
#     <tbody>
#       <tr>
#         <td>빅타임(BIGTIME)</td>
#         <td>BTC, USDT</td>
#         <td>Ethereum</td>
#         <td>공지 게시 시점으로부터 2시간 이내</td>
#         <td>4월 23일 17시 예정</td>
#       </tr>
#       <tr>
#         <td>아카시네트워크(AKT)</td>
#         <td>KRW, BTC, USDT</td>
#         <td>Akash</td>
#         <td>공지 게시 시점으로부터 2시간 이내</td>
#         <td>4월 23일 19시 예정</td>
#       </tr>
#     </tbody>
#   </table>
#
# Parsing Steps:
#   1️⃣ Extract each <tr> block using `_RE_TABLE_ROW`.
#   2️⃣ For each row:
#        - Parse the ticker from parentheses, e.g. "(AKT)" → "AKT"
#        - Split market string, e.g. "KRW, BTC, USDT"
#        - Parse trade time from Korean format ("4월 23일 17시") into ISO-like form.
#   3️⃣ Store each as a separate ListingEntry.
#
# Result:
#   [
#       ListingEntry("BIGTIME", ["BTC", "USDT"], "2024-04-23 17:00"),
#       ListingEntry("AKT", ["KRW", "BTC", "USDT"], "2024-04-23 19:00")
#   ]
#
# Interpretation:
#   - Each <tr> corresponds to **one ticker** → one ListingEntry.
#   - The <table> therefore maps to a collection of multiple listings
#     under the same Upbit announcement.
#
# -----------------------------------------------------------------------------
# 3. Exceptional Cases
# -----------------------------------------------------------------------------
#
# Some notices do not contain a table. In those cases:
#   → The crawler falls back to `_extract_pairs(title)` which parses the title
#     using regex patterns like "(BTC, USDT 마켓)" or "(KRW 마켓)".
#
# Additionally:
#   - When the notice explicitly includes "변경된 거래지원 개시 시점",
#     `_extract_trade_times()` can provide an updated time.
#   - If only specific tickers are delayed (e.g., “AKT의 거래지원 개시 시점을 연기합니다”),
#     the updated time is applied **only to those tickers**, while others retain
#     their original time.
#   - Korean date strings such as "4월 23일 17시 예정" are normalized to
#     "YYYY-MM-DD HH:00" format, with year inferred from `listed_at`.
#
# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
#
# ✅ Each notice may contain one or more tickers.
# ✅ Each <tr> → one asset → one ListingEntry.
# ✅ Title-based parsing only used as a fallback when <table> is absent.
# ✅ Supports both original and updated trade opening timestamps.
# ✅ Handles selective per-ticker delay announcements gracefully.
#
# =============================================================================


SEARCH_API = "https://api-manager.upbit.com/api/v1/announcements/search"
DETAIL_API = "https://api-manager.upbit.com/api/v1/announcements/{id}"
WEB_BASE = "https://upbit.com/service_center/notice?id="


class UpbitNoticeCrawler(NoticeCrawler):
    _RE_PAIR = re.compile(
        r"\((?P<ticker>[A-Z0-9\-]{2,15})\)[^()]*\((?P<markets>[^)]+)\s*마켓\)"
    )
    _RE_TICKERS = re.compile(r"\(([A-Z0-9\-]{2,15})\)")

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

    _RE_TABLE_ROW = re.compile(
        r"<tr>\s*<td>(?P<asset>[^<]+)</td>\s*"
        r"<td>(?P<markets>[^<]*)</td>\s*"
        r"<td>[^<]*</td>\s*"
        r"<td>[^<]*</td>\s*"
        r"<td>(?P<time>[^<]*)</td>",
        re.MULTILINE,
    )

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

            data = self.session.get(
                SEARCH_API,
                params={
                    "os": "web",
                    "page": page,
                    "per_page": page_size,
                    "category": "trade",
                    "search": "신규 거래지원",
                },
            )
            notices = data["data"]["notices"]
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
        seen = set()
        for m in self._RE_PAIR.finditer(title):
            ticker = m.group("ticker")
            markets = [s.strip().upper()
                       for s in m.group("markets").split(",")]
            pairs.append(ListingEntry(ticker=ticker, markets=markets))
            seen.add(ticker)

        tickers = self._RE_TICKERS.findall(title)
        remaining = [t for t in tickers if t not in seen]

        if remaining:
            m2 = re.search(r"\(([^)]+)\s*마켓\)", title)
            if m2:
                markets = [s.strip().upper() for s in m2.group(1).split(",")]
            else:
                markets = Market.from_text(title, self.market_mode)
            for t in remaining:
                pairs.append(ListingEntry(t, markets))
        return pairs

    def _get_year(self, month: str, day: str, hour: str, listed_at: Optional[str]) -> str:
        try:
            base_year = datetime.fromisoformat(
                listed_at).year if listed_at else datetime.now().year
            return f"{base_year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:00"
        except Exception:
            return f"{month}월 {day}일 {hour}시"

    def _parse_table_entries(self, html: str, listed_at: Optional[str]) -> List[ListingEntry]:
        entries: List[ListingEntry] = []
        for m in self._RE_TABLE_ROW.finditer(html):
            asset, markets_text, time_text = (
                m.group("asset"),
                m.group("markets"),
                m.group("time"),
            )
            ticker_match = re.search(r"\(([A-Z0-9\-]{2,15})\)", asset)
            if not ticker_match:
                continue
            ticker = ticker_match.group(1)
            markets = [s.strip().upper()
                       for s in markets_text.split(",") if s.strip()]

            if "월" in time_text:
                m2 = self._RE_TRADE_KR.search(time_text)
                if m2:
                    trade_open = self._get_year(
                        m2.group(1), m2.group(2), m2.group(3), listed_at)
                else:
                    trade_open = time_text
            else:
                trade_open = time_text.strip() or None

            entries.append(ListingEntry(ticker, markets, trade_open))
        return entries

    def _extract_trade_times(
        self, text: str, listed_at: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        trade_open_original, trade_open_updated = None, None
        patterns = [
            ("updated", self._RE_TRADE_UPDATED, None),
            ("original", self._RE_TRADE_PREV, None),
            ("original", self._RE_TRADE_ORIGINAL, None),
            (
                "original",
                self._RE_TRADE_KR,
                lambda m: self._get_year(
                    m.group(1), m.group(2), m.group(3), listed_at),
            ),
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
        """
        Handles selective ticker updates:
        If only specific tickers are delayed (e.g. “AKT의 거래지원 개시 시점을 연기합니다”),
        the updated time is stored separately and does not overwrite the original
        per-ticker time parsed from the table.
        """
        url = DETAIL_API.format(id=notice_id)
        time.sleep(1.0)

        data = self.session.get(url)["data"]
        content_html = data.get("body", "")
        content_text = re.sub(r"<[^>]+>", "", content_html)

        listings = self._parse_table_entries(
            content_html, data.get("listed_at"))
        if not listings:
            listings = hint.listings if hint else self._extract_pairs(
                data["title"])

        trade_open_original, trade_open_updated = self._extract_trade_times(
            content_text, data.get("listed_at")
        )

        preserved_listings: List[ListingEntry] = []
        for entry in listings:
            delayed = re.search(
                rf"[가-힣A-Za-z0-9]*\s*\({re.escape(entry.ticker)}\)\s*의\s*거래지원\s*개시\s*시점을\s*연기",
                content_text,
                flags=re.IGNORECASE,
            )

            original_time = entry.trade_open or trade_open_original
            updated_time = trade_open_updated if delayed else None

            preserved_listings.append(ListingEntry(
                entry.ticker, entry.markets, original_time))

        return ListingDetail(
            id=data["id"],
            title=data["title"],
            url=hint.url if hint else f"{WEB_BASE}{data['id']}",
            listings=preserved_listings,
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
                rows.append(
                    {
                        "ticker": entry.ticker,
                        "markets": ",".join(entry.markets),
                        "trade_open_original": entry.trade_open or d.trade_open_original,
                        "trade_open_updated": d.trade_open_updated,
                        "trade_open": d.trade_open_updated
                        or entry.trade_open
                        or d.trade_open_original,
                    }
                )
        return pd.DataFrame(rows)
