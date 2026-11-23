import time
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. 策略參數 (與 Pine Script 一致)
# ==========================================
MA_LEN1 = 150       # 30週線 (150日線)
MA_LEN2 = 200
ATR_MULT = 0.6
USE_SLOPE = True
SLOPE_LOOK = 10
USE_FAST_TP = True
BIAS_PCT = 35.0
FAST_TP_LEN = 20
S2_STRICT = False   # True = 嚴格模式 (額外要求 150線 > 200線)
BENCHMARK_TICKER = "2800.HK"  # 相對強弱基準 (盈富基金)
BATCH_SIZE = 80

FALLBACK_LIST = {
    "1810.HK": "小米集團-W", "1888.HK": "建滔積層板", "0700.HK": "騰訊控股",
    "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W", "1211.HK": "比亞迪股份",
    "0981.HK": "中芯國際", "2800.HK": "盈富基金", "3033.HK": "南方恒生科技",
    "9660.HK": "地平線機械人", "1686.HK": "新意網集團", "2858.HK": "易鑫集團",
}


def get_ggt_watchlist():
    """直接從港交所披露易 (HKEXnews) 抓取完整港股通名單"""
    url = "https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t=hk"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        watchlist = {}

        for row in soup.select("table.table tbody tr"):
            code_el = row.select_one("td.col-stock-code .mobile-list-body")
            name_el = row.select_one("td.col-stock-name .mobile-list-body")
            if code_el and name_el:
                code_raw = code_el.get_text(strip=True).zfill(5)
                name = name_el.get_text(strip=True)
                yf_ticker = (code_raw[1:] if code_raw.startswith("0") else code_raw) + ".HK"
                watchlist[yf
