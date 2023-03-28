#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import *
from alpaca.trading.client import TradingClient

def main():
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = trading_client.get_account()
    for property_name, value in account:
        print(f"\"{property_name}\": {value}")


if __name__ == '__main__':
    main()