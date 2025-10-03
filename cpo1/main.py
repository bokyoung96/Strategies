from dataclasses import asdict
from typing import Optional

from cpo1.market import Market
from cpo1.session import ApiSession
from cpo1.upbit import UpbitNoticeCrawler


def get_upbit_new_listings(market_mode: Market = Market.ALL,
                           limit: int = 10,
                           start_page: int = 1,
                           end_page: Optional[int] = None):
    session = ApiSession()
    crawler = UpbitNoticeCrawler(session=session,
                                 market_mode=market_mode,
                                 limit=limit,
                                 start_page=start_page,
                                 end_page=end_page)

    return {
        "summaries": [asdict(s) for s in crawler.summaries],
        "details": [asdict(d) for d in crawler.details],
        "crawled": crawler.crawled()
    }
