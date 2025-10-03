import os
import sys
from curses import start_color

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cpo1.main import get_upbit_new_listings
from cpo1.market import Market

if __name__ == "__main__":
    data = get_upbit_new_listings(max_items=100,
                                  market_mode=Market.ALL,
                                  start_page=1,
                                  end_page=None)
