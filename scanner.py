import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. 股票監察名單 (可自行增減港股代號)
# ==========================================
WATCHLIST = {
    "1810.HK": "小米集團-W",
    "1888.HK": "建滔積層板",
    "0700.HK": "騰訊控股",
    "9988.HK": "阿里巴巴-W",
    "3690.HK": "美團-W",
    "1211.HK": "比亞迪股份",
    "0981.HK": "中芯國際",
    "2800.HK": "盈富基金",
    "3033.HK": "南方恒生科技",
    "9660.HK": "地平線機械人",
    "1686.HK": "新意網集團",
    "2858.HK": "易鑫集團",
}

# ==========================================
# 2. 策略參數 (與你的 Pine Script 完全一致)
# ==========================================
MA_LEN1 = 150
MA_LEN2 = 200
ATR_MULT = 0.6
USE_SLOPE = True
SLOPE_LOOK = 10
USE_FAST_TP = True
BIAS_PCT = 35.0
FAST_TP_LEN = 20


def analyze_stock(ticker, name):
    try:
        # 抓取 2 年日線資料，確保 200 天均線計算準確
        df = yf.Ticker(ticker).history(period="2y")
        if df.empty or len(df) < MA_LEN2 + 15:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # 計算 150 / 200 SMA 與 20 EMA
        df["ma1"] = close.rolling(MA_LEN1).mean()
        df["ma2"] = close.rolling(MA_LEN2).mean()
        df["ema_tp"] = close.ewm(span=FAST_TP_LEN, adjust=False).mean()

        # 計算 ATR(14) (Wilder's smoothing，對齊 TradingView ta.atr)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # 支持/阻力帶與緩衝區
        df["band_top"] = np.maximum(df["ma1"], df["ma2"])
        df["band_bot"] = np.minimum(df["ma1"], df["ma2"])
        df["buy_thresh"] = df["band_top"] + df["atr"] * ATR_MULT
        df["sell_thresh"] = df["band_bot"] - df["atr"] * ATR_MULT
        df["slope_ok"] = df["ma1"] >= df["ma1"].shift(SLOPE_LOOK) if USE_SLOPE else True
        df["bias"] = (close - df["ma1"]) / df["ma1"] * 100

        # 逐日執行狀態機 (完全對齊 Pine Script 邏輯)
        trend = 0
        high_bias_hit = False
        last_signal = "無"
        last_signal_date = "-"

        for i in range(MA_LEN2, len(df)):
            c = df["Close"].iloc[i]
            prev_c = df["Close"].iloc[i - 1]
            ema_curr = df["ema_tp"].iloc[i]
            ema_prev = df["ema_tp"].iloc[i - 1]
            b_thresh = df["buy_thresh"].iloc[i]
            s_thresh = df["sell_thresh"].iloc[i]
            s_ok = df["slope_ok"].iloc[i]
            bias_val = df["bias"].iloc[i]
            dt_str = df.index[i].strftime("%Y-%m-%d")

            if trend == 1 and bias_val >= BIAS_PCT:
                high_bias_hit = True

            buy_cond = (c > b_thresh) and s_ok and (trend != 1)
            normal_sell = (c < s_thresh) and (trend == 1)
            crossunder_ema = (prev_c >= ema_prev) and (c < ema_curr)
            fast_tp_sell = USE_FAST_TP and (trend == 1) and high_bias_hit and crossunder_ema

            if buy_cond:
                trend = 1
                high_bias_hit = False
                last_signal = "買入"
                last_signal_date = dt_str
            elif normal_sell:
                trend = -1
                high_bias_hit = False
                last_signal = "賣出"
                last_signal_date = dt_str
            elif fast_tp_sell:
                trend = -1
                high_bias_hit = False
                last_signal = "急升止盈"
                last_signal_date = dt_str

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        chg_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        return {
            "ticker": ticker,
            "name": name,
            "price": latest["Close"],
            "chg_pct": chg_pct,
            "trend": trend,
            "high_bias_hit": high_bias_hit,
            "bias": latest["bias"],
            "band_top": latest["band_top"],
            "band_bot": latest["band_bot"],
            "ema_tp": latest["ema_tp"],
            "last_signal": last_signal,
            "last_signal_date": last_signal_date,
            "is_today_signal": (last_signal_date == latest_date),
        }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def generate_html(results):
    hkt = timezone(timedelta(hours=8))
    now_str = datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT")

    # 排序：今日有訊號優先 -> 多頭持倉優先 -> 乖離率高至低
    results.sort(key=lambda x: (not x["is_today_signal"], x["trend"] != 1, -x["bias"]))

    rows_html = ""
    for r in results:
        chg_color = "#00c853" if r["chg_pct"] >= 0 else "#ff1744"
        chg_sign = "+" if r["chg_pct"] >= 0 else ""

        if r["trend"] == 1:
            status_badge = '<span class="badge green">🟢 多頭持倉</span>'
        elif r["trend"] == -1:
            status_badge = '<span class="badge red">🔴 空頭/觀望</span>'
        else:
            status_badge = '<span class="badge gray">⚪ 觀察中</span>'

        # 急升止盈保護提示
        tp_info = f'<span class="tp-active">🔥 已啟動 ({r["ema_tp"]:.2f})</span>' if (r["trend"] == 1 and r["high_bias_hit"]) else "-"

        # 最近訊號標籤
        sig_class = "green" if r["last_signal"] == "買入" else ("orange" if r["last_signal"] == "急升止盈" else "red")
        today_tag = ' <span class="new-tag">NEW</span>' if r["is_today_signal"] else ""
        sig_html = f'<span class="badge {sig_class}">{r["last_signal"]}</span>{today_tag}<div class="sub-date">{r["last_signal_date"]}</div>'

        rows_html += f"""
        <tr>
            <td><strong>{r['ticker']}</strong><div class="sub-name">{r['name']}</div></td>
            <td><strong>{r['price']:.2f}</strong><div style="color:{chg_color};font-size:12px;">{chg_sign}{r['chg_pct']:.2f}%</div></td>
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
    <title>150/200 支持阻力帶掃描 Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 15px; }}
        h1 {{ margin: 0; font-size: 22px; color: #f8fafc; }}
        .meta {{ font-size: 13px; color: #94a3b8; }}
        .table-wrap {{ overflow-x: auto; background: #1e293b; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th, td {{ padding: 14px 16px; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 13px; }}
        tr:hover {{ background: #273549; }}
        .sub-name, .sub-date {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
        .badge {{ padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; display: inline-block; }}
        .badge.green {{ background: rgba(0, 200, 83, 0.2); color: #00e676; border: 1px solid #00c853; }}
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
                <h1>📊 150/200 支持阻力帶 (抗洗盤 + 急升止盈)</h1>
                <div class="meta">參數：SMA 150/200 | 緩衝 0.6 ATR | 10日斜率 | 35% 乖離啟動 20 EMA 止盈</div>
            </div>
            <div class="meta">最後更新：{now_str}</div>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>股票</th>
                        <th>現價 / 升跌</th>
                        <th>目前趨勢</th>
                        <th>最近訊號</th>
                        <th>150/200 支持阻力帶</th>
                        <th>150線乖離</th>
                        <th>20 EMA 急升止盈</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    data = []
    for ticker, name in WATCHLIST.items():
        res = analyze_stock(ticker, name)
        if res:
            data.append(res)
    generate_html(data)
    print(f"Done! Generated index.html with {len(data)} stocks.")
