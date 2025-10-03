import re
from typing import List

from cpo1.base import NoticeCrawler
from cpo1.market import Market
from cpo1.models import ListingEntry, NoticeSummary
from cpo1.session import ApiSession

API_BASE = "https://api-manager.upbit.com/api/v1/announcements"
WEB_BASE = "https://upbit.com/service_center/notice?id="


class UpbitNoticeCrawler(NoticeCrawler):
    _RE_NEW_LISTING = re.compile(r"신규\s*거래\s*지원")
    _RE_PAIR = re.compile(
        r"\((?P<ticker>[A-Z0-9\-]{2,15})\)\((?P<markets>[A-Z ,]+)\s*마켓\)"
    )

    def __init__(self, session: ApiSession, market_mode: Market = Market.KRW):
        self.session = session
        self.market_mode = market_mode

    def _extract_pairs(self, title: str) -> List[ListingEntry]:
        pairs: List[ListingEntry] = []

        for m in self._RE_PAIR.finditer(title):
            ticker = m.group("ticker")
            markets = [s.strip().upper()
                       for s in m.group("markets").split(",")]
            pairs.append(ListingEntry(ticker=ticker, markets=markets))

        if not pairs:
            tickers = re.findall(r"\(([A-Z0-9\-]{2,15})\)", title)
            markets = Market.from_text(title, self.market_mode)
            for t in tickers:
                pairs.append(ListingEntry(ticker=t, markets=markets))
        return pairs

    def iter_new_listings(self, limit: int = 50) -> List[NoticeSummary]:
        results: List[NoticeSummary] = []
        page = 1

        while len(results) < limit:
            data = self.session.get(API_BASE, params={
                "os": "web",
                "page": page,
                "per_page": 20,
                "category": "trade",
            })
            notices = data["data"]["notices"]
            if not notices:
                break

            for n in notices:
                title = n["title"]
                if not self._RE_NEW_LISTING.search(title):
                    continue

                url = f"{WEB_BASE}{n['id']}"
                listings = self._extract_pairs(title)

                results.append(
                    NoticeSummary(
                        id=n["id"],
                        title=title,
                        url=url,
                        listings=listings,
                    )
                )
                if len(results) >= limit:
                    break
            page += 1
        return results
