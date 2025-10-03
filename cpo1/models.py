from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ListingEntry:
    ticker: str
    markets: List[str]


@dataclass(frozen=True)
class NoticeSummary:
    id: Optional[int]
    title: str
    url: str
    listings: List[ListingEntry]


@dataclass(frozen=True)
class ListingDetail:
    id: Optional[int]
    title: str
    url: str
    listings: List[ListingEntry]

    trade_open_original: Optional[str]
    trade_open_updated: Optional[str]

    content_text: str

    category: Optional[str] = None
    need_new_badge: Optional[bool] = None
    need_update_badge: Optional[bool] = None
    listed_at: Optional[str] = None
    first_listed_at: Optional[str] = None
