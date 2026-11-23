import os
import json
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. Strategy Parameters
# ==========================================
MA_LEN1 = 150       # 30-week MA (150-day MA)
MA_LEN2 = 200
ATR_MULT = 0.6
USE_SLOPE = True
SLOPE_LOOK = 10
USE_FAST_TP = True
BIAS_PCT = 35.0
FAST_TP_LEN = 20
S2_STRICT = False
BENCHMARK_TICKER = "2800.HK"
BATCH_SIZE = 80
SECTOR_CACHE_FILE = "sector_cache.json"

SECTOR_ZH = {
    "Technology": "科技",
    "Financial Services": "金融",
    "Consumer Cyclical": "非必需消費",
    "Consumer Defensive": "必需消費",
    "Healthcare": "醫療保健",
    "Industrials": "工業",
    "Communication Services": "電訊及媒體",
    "Real Estate": "地產",
    "Energy": "能源",
    "Utilities": "公用事業",
    "Basic Materials": "原材料",
}

FALLBACK_LIST = {
    "1810.HK": "小米集團-W", "1888.HK": "建滔積層板", "0700.HK": "騰訊控股",
    "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W", "1211.HK": "比亞迪股份",
    "0981.HK": "中芯國際", "2800.HK": "盈富基金", "3033.HK": "南方恒生科技",
    "9660.HK": "地平線機械人", "1686.HK": "新意網集團", "2858.HK": "易鑫集團",
}


def get_ggt_watchlist():
    """Fetch full HKEX Southbound (GGT) stock list directly from HKEXnews."""
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
                watchlist[yf_ticker] = name

        if len(watchlist) > 50:
            print(f"Fetched {len(watchlist)} GGT stocks from HKEXnews.")
            return watchlist
        raise ValueError(f"Unexpected stock count: {len(watchlist)}")
    except Exception as e:
        print(f"HKEXnews fetch failed ({e}), using fallback list.")
        return FALLBACK_LIST


def load_and_update_sectors(tickers):
    """Load cached sectors and fetch missing ones concurrently."""
    cache = {}
    if os.path.exists(SECTOR_CACHE_FILE):
        try:
            with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f"Fetching sector info for {len(missing)} stocks...")

        def fetch_one(t):
            try:
                info = yf.Ticker(t).info
                sec = info.get("sector", "Other")
                return t, sec if sec else "Other"
            except Exception:
                return t, "Other"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_one, t) for t in missing]
            for fut in as_completed(futures):
                t, sec = fut.result()
                cache[t] = sec

    with open(SECTOR_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return cache


def get_benchmark_series():
    """Download benchmark (2800.HK) close prices."""
    try:
        df_b = yf.Ticker(BENCHMARK_TICKER).history(period="2y")
        s = df_b["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception as e:
        print(f"Failed to download benchmark {BENCHMARK_TICKER}: {e}")
        return pd.Series(dtype=float)


def analyze_dataframe(ticker, name, sector_en, df, bench_close):
    """Run 150/200 Support/Resistance Band + Fast TP + Stage 2 + RS + vs 2800.HK."""
    try:
        df = df.dropna(subset=["Close", "High", "Low"]).copy()
        if len(df) < MA_LEN2 + 15:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        df["ma1"] = close.rolling(MA_LEN1).mean()
        df["ma2"] = close.rolling(MA_LEN2).mean()
        df["ema_tp"] = close.ewm(span=FAST_TP_LEN, adjust=False).mean()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

        df["band_top"] = np.maximum(df["ma1"], df["ma2"])
        df["band_bot"] = np.minimum(df["ma1"], df["ma2"])
        df["buy_thresh"] = df["band_top"] + df["atr"] * ATR_MULT
        df["sell_thresh"] = df["band_bot"] - df["atr"] * ATR_MULT
        df["slope_up"] = df["ma1"] > df["ma1"].shift(SLOPE_LOOK)
        df["slope_ok"] = df["ma1"] >= df["ma1"].shift(SLOPE_LOOK) if USE_SLOPE else True
        df["bias"] = (close - df["ma1"]) / df["ma1"] * 100

        if S2_STRICT:
            df["is_stage2"] = (close > df["ma1"]) & df["slope_up"] & (df["ma1"] > df["ma2"])
        else:
            df["is_stage2"] = (close > df["ma1"]) & df["slope_up"]

        trend = 0
        high_bias_hit = False
        last_signal = "無"
        last_signal_date = "-"

        closes = df["Close"].values
        emas = df["ema_tp"].values
        b_threshs = df["buy_thresh"].values
        s_threshs = df["sell_thresh"].values
        s_oks = df["slope_ok"].values
        biases = df["bias"].values
        dates = df.index.strftime("%Y-%m-%d").values

        for i in range(MA_LEN2, len(df)):
            c = closes[i]
            prev_c = closes[i - 1]
            ema_curr = emas[i]
            ema_prev = emas[i - 1]

            if trend == 1 and biases[i] >= BIAS_PCT:
                high_bias_hit = True

            buy_cond = (c > b_threshs[i]) and s_oks[i] and (trend != 1)
            normal_sell = (c < s_threshs[i]) and (trend == 1)
            crossunder_ema = (prev_c >= ema_prev) and (c < ema_curr)
            fast_tp_sell = USE_FAST_TP and (trend == 1) and high_bias_hit and crossunder_ema

            if buy_cond:
                trend = 1
                high_bias_hit = False
                last_signal = "買入"
                last_signal_date = dates[i]
            elif normal_sell:
                trend = -1
                high_bias_hit = False
                last_signal = "賣出"
                last_signal_date = dates[i]
            elif fast_tp_sell:
                trend = -1
                high_bias_hit = False
                last_signal = "急升止盈"
                last_signal_date = dates[i]

        # Weighted RS raw score (3M*40% + 6M*20% + 9M*20% + 12M*20%)
        n = len(closes)
        c_now = closes[-1]
        r3 = (c_now / closes[-63] - 1) if n >= 64 else 0.0
        r6 = (c_now / closes[-126] - 1) if n >= 127 else r3
        r9 = (c_now / closes[-189] - 1) if n >= 190 else r6
        r12 = (c_now / closes[-252] - 1) if n >= 253 else r9
        raw_rs_score = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12

        # Mansfield RS (vs 2800.HK 200-day relative strength %)
        mansfield_rs = 0.0
        if not bench_close.empty:
            stock_s = df["Close"].copy()
            stock_s.index = pd.to_datetime(stock_s.index).tz_localize(None).normalize()
            aligned = pd.concat([stock_s.rename("s"), bench_close.rename("b")], axis=1).dropna()
            if len(aligned) >= 200:
                rp = aligned["s"] / aligned["b"]
                rp_ma = rp.rolling(200).mean()
                if rp_ma.iloc[-1] > 0:
                    mansfield_rs = float((rp.iloc[-1] / rp_ma.iloc[-1] - 1) * 100)

        latest_price = closes[-1]
        prev_price = closes[-2]
        chg_pct = (latest_price - prev_price) / prev_price * 100
        latest_date = dates[-1]
        is_stage2_now = bool(df["is_stage2"].iloc[-1])

        return {
            "ticker": ticker,
            "name": name,
            "sector_zh": SECTOR_ZH.get(sector_en, "其他"),
            "price": float(latest_price),
            "chg_pct": float(chg_pct),
            "trend": trend,
            "is_stage2": is_stage2_now,
            "high_bias_hit": high_bias_hit,
            "bias": float(biases[-1]),
            "band_top": float(df["band_top"].iloc[-1]),
            "band_bot": float(df["band_bot"].iloc[-1]),
            "ema_tp": float(emas[-1]),
            "last_signal": last_signal,
            "last_signal
