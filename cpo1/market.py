import re
from enum import Enum, auto
from typing import List


class Market(Enum):
    KRW = auto()
    BTC = auto()
    USDT = auto()
    ALL = auto()
    OTHER = auto()

    @staticmethod
    def from_text(text: str, mode: "Market" = None) -> List[str]:
        matches = re.findall(r"([A-Z]{2,10})\s*마켓", text, re.IGNORECASE)
        uniq = []
        for m in {s.upper() for s in matches}:
            if mode == Market.ALL:
                uniq.append(m)
            else:
                try:
                    if Market[m] == mode or m in ["KRW", "BTC", "USDT"]:
                        uniq.append(m)
                except KeyError:
                    uniq.append("OTHER")
        return uniq
