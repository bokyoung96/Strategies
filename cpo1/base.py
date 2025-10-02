from typing import Iterable, Optional, Protocol

from cpo1.models import ListingDetail, NoticeSummary


class NoticeCrawler(Protocol):
    def iter_new_listings(self, limit: int = 50) -> Iterable[NoticeSummary]:
        ...

    def fetch_detail(
        self, url: str, hint: Optional[NoticeSummary] = None
    ) -> ListingDetail:
        ...
