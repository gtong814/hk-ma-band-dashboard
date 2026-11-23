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
# 1. 策略參數
# ==========================================
MA_LEN1 = 150       # 30週線 (150日線)
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

# 行業英文轉繁體中文對照表
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
                watchlist[yf_ticker] = name

        if len(watchlist) > 50:
            print(f"成功從港交所獲取港股通名單，共 {len(watchlist)} 隻股票。")
            return watchlist
        else:
            raise ValueError(f"抓取數量異常 ({len(watchlist)})")
    except Exception as e:
        print(f"從港交所獲取名單失敗 ({e})，改用備用名單。")
        return FALLBACK_LIST


def load_and_update_sectors(tickers):
    """讀取本地行業快取，並多線程補抓新股票的行業分類"""
    cache = {}
    if os.path.exists(SECTOR_CACHE_FILE):
        try:
            with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f"需要查詢 {len(missing)} 隻股票的行業分類 (首次執行需時約 1-2 分鐘)...")

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
    """下載基準指數 (2800.HK) 收市價"""
    try:
        df_b = yf.Ticker(BENCHMARK_TICKER).history(period="2y")
        s = df_b["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception as e:
        print(f"下載基準 {BENCHMARK_TICKER} 失敗: {e}")
        return pd.Series(dtype=float)


def analyze_dataframe(ticker, name, sector_en, df, bench_close):
    """套用 150/200 支持阻力帶 + 急升止盈 + Stage 2 + RS + vs 2800.HK"""
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
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

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

        # 個股加權 RS 原始分數 (3M*40% + 6M*20% + 9M*20% + 12M*20%)
        n = len(closes)
        c_now = closes[-1]
        r3 = (c_now / closes[-63] - 1) if n >= 64 else 0.0
        r6 = (c_now / closes[-126] - 1) if n >= 127 else r3
        r9 = (c_now / closes[-189] - 1) if n >= 190 else r6
        r12 = (c_now / closes[-252] - 1) if n >= 253 else r9
        raw_rs_score = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12

        # Mansfield RS (vs 2800.HK 的 200日相對強弱 %)
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
            "last_signal_date": last_signal_date,
            "is_today_signal": (last_signal_date == latest_date),
            "raw_rs_score": float(raw_rs_score),
            "mansfield_rs": mansfield_rs,
            "rs_rating": 50,
            "sector_rs": 50,
        }
    except Exception:
        return None


def scan_all_stocks(watchlist):
    tickers = list(watchlist.keys())
    sector_map = load_and_update_sectors(tickers)
    bench_close = get_benchmark_series()
    results = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"正在下載第 {i+1} ~ {min(i+BATCH_SIZE, len(tickers))} 隻股票...")
        try:
            data = yf.download(batch, period="2y", group_by="ticker", threads=True, progress=False)
            for t in batch:
                if len(batch) == 1:
                    df_t = data
                elif t in data.columns.get_level_values(0):
                    df_t = data[t]
                else:
                    continue
                res = analyze_dataframe(t, watchlist[t], sector_map.get(t, "Other"), df_t, bench_close)
                if res:
                    results.append(res)
        except Exception as e:
            print(f"批次下載發生錯誤: {e}")
        time.sleep(1)

    if results:
        # 1. 計算個股 RS Rating (1-99 百分位)
        scores = pd.Series([r["raw_rs_score"] for r in results])
        ranks = scores.rank(pct=True)
        for idx, r in enumerate(results):
            r["rs_rating"] = int(max(1, min(99, round(ranks.iloc[idx] * 99))))

        # 2. 計算行業 RS (依各行業平均個股 RS 評分排名 1-99)
        df_sec = pd.DataFrame([{"sec": r["sector_zh"], "rs": r["rs_rating"]} for r in results])
        sec_means = df_sec.groupby("sec")["rs"].mean()
        sec_ranks = sec_means.rank(pct=True)
        sec_rs_dict = {sec: int(max(1, min(99, round(val * 99)))) for sec, val in sec_ranks.items()}

        for r in results:
            r["sector_rs"] = sec_rs_dict.get(r["sector_zh"], 50)

    return results


def generate_html(results):
    hkt = timezone(timedelta(hours=8))
    now_str = datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT")

    total_cnt = len(results)
    today_buy = sum(1 for r in results if r["is_today_signal"] and r["last_signal"] == "買入")
    today_sell = sum(1 for r in results if r["is_today_signal"] and r["last_signal"] in ["賣出", "急升止盈"])
    bull_cnt = sum(1 for r in results if r["trend"] == 1)
    s2_cnt = sum(1 for r in results if r["is_stage2"])
    rs80_cnt = sum(1 for r in results if r["rs_rating"] >= 80)

    # 排序：今日訊號 -> Stage 2 -> 個股 RS 由高至低
    results.sort(key=lambda x: (not x["is_today_signal"], not x["is_stage2"], -x["rs_rating"]))

    rows_html = ""
    for r in results:
        chg_color = "#00c853" if r["chg_pct"] >= 0 else "#ff1744"
        chg_sign = "+" if r["chg_pct"] >= 0 else ""

        status_key = "bull" if r["trend"] == 1 else ("bear" if r["trend"] == -1 else "neutral")
        if r["trend"] == 1:
            status_badge = '<span class="badge green">🟢 多頭持倉</span>'
        elif r["trend"] == -1:
            status_badge = '<span class="badge red">🔴 空頭/觀望</span>'
        else:
            status_badge = '<span class="badge gray">⚪ 觀察中</span>'

        row_class = "stage2-row" if r["is_stage2"] else ""
        s2_badge = '<span class="badge s2">🌿 Stage 2</span>' if r["is_stage2"] else '<span style="color:#64748b;">-</span>'

        # 個股 RS 評分
        rs_val = r["rs_rating"]
        rs_class = "green" if rs_val >= 80 else ("blue" if rs_val >= 60 else "gray")
        rs_html = f'<span class="badge {rs_class}">RS {rs_val}</span>'

        # 獨立欄位：vs 2800.HK (Mansfield RS %)
        mrs_val = r["mansfield_rs"]
        mrs_color = "#00e676" if mrs_val >= 0 else "#ff5252"
        mrs_html = f'<strong style="color:{mrs_color};">{mrs_val:+.1f}%</strong>'

        # 獨立欄位：行業 / 行業 RS
        sec_rs = r["sector_rs"]
        sec_class = "green" if sec_rs >= 70 else ("blue" if sec_rs >= 40 else "gray")
        sec_html = f'<div>{r["sector_zh"]}</div><span class="badge {sec_class}" style="margin-top:3px;">行業 RS {sec_rs}</span>'

        tp_active = r["trend"] == 1 and r["high_bias_hit"]
        tp_info = f'<span class="tp-active">🔥 已啟動 ({r["ema_tp"]:.2f})</span>' if tp_active else "-"

        sig_class = "green" if r["last_signal"] == "買入" else ("orange" if r["last_signal"] == "急升止盈" else "red")
        today_tag = ' <span class="new-tag">NEW</span>' if r["is_today_signal"] else ""
        sig_html = f'<span class="badge {sig_class}">{r["last_signal"]}</span>{today_tag}<div class="sub-date">{r["last_signal_date"]}</div>'

        rows_html += f"""
        <tr class="{row_class}" data-today="{'1' if r['is_today_signal'] else '0'}" data-status="{status_key}" data-s2="{'1' if r['is_stage2'] else '0'}" data-rs80="{'1' if rs_val >= 80 else '0'}" data-tp="{'1' if tp_active else '0'}" data-search="{r['ticker'].lower()} {r['name'].lower()} {r['sector_zh'].lower()}">
            <td><strong>{r['ticker']}</strong><div class="sub-name">{r['name']}</div></td>
            <td><strong>{r['price']:.2f}</strong><div style="color:{chg_color};font-size:12px;">{chg_sign}{r['chg_pct']:.2f}%</div></td>
            <td>{rs_html}</td>
            <td>{mrs_html}</td>
            <td>{sec_html}</td>
            <td>{s2_badge}</td>
            <td>{status_badge}</td>
            <td>{sig_html}</td>
            <td>{r['band_bot']:.2f} ~ {r['band_top']:.2f}</td>
            <td>{r['bias']:+.1f}%</td>
            <td>{tp_info}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股通 150/200 支持阻力帶 + Stage 2 + 行業RS Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1380px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 12px; }}
        h1 {{ margin: 0; font-size: 22px; color: #f8fafc; }}
        .meta {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
        .card {{ background: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155; }}
        .card-title {{ font-size: 12px; color: #94a3b8; }}
        .card-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
        .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; align-items: center; }}
        input[type="text"] {{ background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 240px; }}
        .btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #475569; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }}
        .btn.active, .btn:hover {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
        .table-wrap {{ overflow-x: auto; background: #1e293b; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th, td {{ padding: 12px 12px; border-bottom: 1px solid #334155; font-size: 13px; white-space: nowrap; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 12px; position: sticky; top: 0; }}
        tr:hover {{ background: #273549; }}
        tr.stage2-row {{ background: rgba(0, 200, 83, 0.10); }}
        tr.stage2-row:hover {{ background: rgba(0, 200, 83, 0.18); }}
        .sub-name, .sub-date {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
        .badge {{ padding: 3px 7px; border-radius: 6px; font-size: 12px; font-weight: 600; display: inline-block; }}
        .badge.green {{ background: rgba(0, 200, 83, 0.2); color: #00e676; border: 1px solid #00c853; }}
        .badge.blue {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }}
        .badge.s2 {{ background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; }}
        .badge.red {{ background: rgba(255, 23, 68, 0.2); color: #ff5252; border: 1px solid #ff1744; }}
        .badge.orange {{ background: rgba(255, 152, 0, 0.2); color: #ffb74d; border: 1px solid #ff9800; }}
        .badge.gray {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}
        .new-tag {{ background: #eab308; color: #000; font-size: 10px; font-weight: 800; padding: 2px 5px; border-radius: 4px; margin-left: 4px; vertical-align: middle; }}
        .tp-active {{ color: #ffb74d; font-weight: 600; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>📊 港股通 150/200 支持阻力帶 + Stage 2 + 行業 RS 掃描</h1>
                <div class="meta">每日 12:15 & 16:20 HKT 自動更新 | 綠色底色 = Stage 2 上升期 | 搜尋欄支援直接輸入行業（如「科技」、「醫療」）</div>
            </div>
            <div class="meta">最後更新：{now_str}</div>
        </div>

        <div class="stats">
            <div class="card"><div class="card-title">掃描港股通總數</div><div class="card-val">{total_cnt}</div></div>
            <div class="card"><div class="card-title">🌿 Stage 2 上升期</div><div class="card-val" style="color:#34d399;">{s2_cnt}</div></div>
            <div class="card"><div class="card-title">💪 強勢股 (RS ≥ 80)</div><div class="card-val" style="color:#00e676;">{rs80_cnt}</div></div>
            <div class="card"><div class="card-title">今日新買入訊號</div><div class="card-val" style="color:#00e676;">{today_buy}</div></div>
            <div class="card"><div class="card-title">今日賣出 / 止盈</div><div class="card-val" style="color:#ff5252;">{today_sell}</div></div>
        </div>

        <div class="controls">
            <input type="text" id="searchInput" placeholder="🔍 搜尋代號、名稱或行業 (如 科技)..." onkeyup="filterTable()">
            <button class="btn active" onclick="setFilter('all', this)">全部 ({total_cnt})</button>
            <button class="btn" onclick="setFilter('today', this)">⭐ 今日新訊號 ({today_buy + today_sell})</button>
            <button class="btn" onclick="setFilter('s2', this)">🌿 Stage 2 ({s2_cnt})</button>
            <button class="btn" onclick="setFilter('rs80', this)">💪 RS ≥ 80 ({rs80_cnt})</button>
            <button class="btn" onclick="setFilter('bull', this)">🟢 多頭持倉 ({bull_cnt})</button>
            <button class="btn" onclick="setFilter('tp', this)">🔥 急升止盈啟動</button>
        </div>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>股票</th>
                        <th>現價 / 升跌</th>
                        <th>個股 RS</th>
                        <th>vs 2800.HK</th>
                        <th>行業 / 行業 RS</th>
                        <th>Weinstein 階段</th>
                        <th>目前趨勢</th>
                        <th>最近訊號</th>
                        <th>150/200 支持阻力帶</th>
                        <th>150線乖離</th>
                        <th>20 EMA 急升止盈</th>
                    </tr>
                </thead>
                <tbody id="stockTableBody">
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let currentFilter = 'all';
        function setFilter(type, btn) {{
            currentFilter = type;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterTable();
        }}
        function filterTable() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            const rows = document.querySelectorAll('#stockTableBody tr');
            rows.forEach(row => {{
                const matchSearch = row.getAttribute('data-search').includes(query);
                let matchFilter = true;
                if (currentFilter === 'today') matchFilter = row.getAttribute('data-today') === '1';
                else if (currentFilter === 's2') matchFilter = row.getAttribute('data-s2') === '1';
                else if (currentFilter === 'rs80') matchFilter = row.getAttribute('data-rs80') === '1';
                else if (currentFilter === 'bull') matchFilter = row.getAttribute('data-status') === 'bull';
                else if (currentFilter === 'tp') matchFilter = row.getAttribute('data-tp') === '1';
                row.style.display = (matchSearch && matchFilter) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    watchlist = get_ggt_watchlist()
    results = scan_all_stocks(watchlist)
    generate_html(results)
    print(f"完成！共分析 {len(results)} 隻港股通股票並生成 index.html。")
