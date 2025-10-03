from dataclasses import asdict

from cpo1.market import Market
from cpo1.session import ApiSession
from cpo1.upbit import UpbitNoticeCrawler


def get_upbit_new_listings(max_items: int = 10, market_mode: Market = Market.ALL):
    sess = ApiSession()
    crawler = UpbitNoticeCrawler(sess, market_mode, limit=max_items)

    return {
        "summaries": [asdict(s) for s in crawler.summaries],
        "details": [asdict(d) for d in crawler.details],
        "crawled": crawler.crawled()
    }
