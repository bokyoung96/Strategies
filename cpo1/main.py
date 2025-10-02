import json
from dataclasses import asdict
from typing import Dict, List

from cpo1.market import Market
from cpo1.session import BrowserSession
from cpo1.upbit import UpbitNoticeCrawler


def get_upbit_new_listings(max_items: int = 10, headless: bool = True, market_mode: Market = None) -> Dict[str, List[dict]]:
    with BrowserSession(headless=headless) as sess:
        crawler = UpbitNoticeCrawler(session=sess, market_mode=market_mode)
        summaries = list(crawler.iter_new_listings(limit=max_items))
        return {
            "summaries": [asdict(s) for s in summaries],
        }
