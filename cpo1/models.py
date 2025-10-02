from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class NoticeSummary:
    id: Optional[int]
    title: str
    url: str
    ticker: Optional[str]
    markets: List[str]


@dataclass(frozen=True)
class ListingDetail:
    id: Optional[int]
    title: str
    url: str
    ticker: Optional[str]
    markets: List[str]
    trade_open_text: Optional[str]
    deposit_open_text: Optional[str]
    withdraw_open_text: Optional[str]
    content_text: str
