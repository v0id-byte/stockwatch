"""技术指标分析（纯 pandas，不调 LLM）"""
import numpy as np
import pandas as pd
from loguru import logger


def compute_tech_score(df_kline: list[dict]) -> dict:
    """
    对日线 K 线计算技术指标，返回 (score, details)
    score ∈ [-1, +1]，正值看多，负值看空
    """
    if not df_kline or len(df_kline) < 20:
        return {"score": 0, "details": {"error": "数据不足"}}

    df = pd.DataFrame(df_kline)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    n = len(df)
    score = 0.0
    details = {}

    # ---- 均线 ----
    for window in [5, 10, 20, 60]:
        if n >= window:
            df[f"ma{window}"] = df["close"].rolling(window).mean()
            details[f"ma{window}"] = round(df[f"ma{window}"].iloc[-1], 2)

    # 金叉死叉
    if n >= 20:
        ma5_above_ma20 = df["ma5"].iloc[-1] > df["ma20"].iloc[-1] if n >= 20 else False
        ma10_above_ma20 = df["ma10"].iloc[-1] > df["ma20"].iloc[-1] if n >= 20 else False
        if ma5_above_ma20 and ma10_above_ma20:
            score += 0.2
            details["ma_cross"] = "golden_cross"
        elif not ma5_above_ma20 and not ma10_above_ma20:
            score -= 0.2
            details["ma_cross"] = "death_cross"

    # ---- MACD ----
    if n >= 26:
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd_bar = (dif - dea) * 2
        details["macd_dif"] = round(dif.iloc[-1], 3)
        details["macd_dea"] = round(dea.iloc[-1], 3)
        details["macd_bar"] = round(macd_bar.iloc[-1], 3)
        if macd_bar.iloc[-1] > 0:
            score += 0.15
        else:
            score -= 0.15

    # ---- RSI(14) ----
    if n >= 15:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        details["rsi14"] = round(rsi_val, 1)
        if rsi_val > 70:
            score -= 0.15
        elif rsi_val < 30:
            score += 0.15
        else:
            score += 0.05 if rsi_val > 50 else -0.05

    # ---- KDJ ----
    if n >= 20:
        low14 = df["low"].rolling(9).min()
        high14 = df["high"].rolling(9).max()
        rsv = (df["close"] - low14) / (high14 - low14 + 1e-9)
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        details["kdj_k"] = round(k.iloc[-1], 2)
        details["kdj_d"] = round(d.iloc[-1], 2)
        details["kdj_j"] = round(j.iloc[-1], 2)
        if j.iloc[-1] > 80:
            score -= 0.1
        elif j.iloc[-1] < 20:
            score += 0.1

    # ---- 布林带(20,2) ----
    if n >= 20:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        bbp = (df["close"].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)
        details["boll_position"] = round(bbp, 2)
        if bbp > 0.9:
            score -= 0.1
        elif bbp < 0.1:
            score += 0.1

    # ---- 量比 ----
    vol_avg5 = df["volume"].rolling(5).mean().iloc[-1]
    vol_now = df["volume"].iloc[-1]
    if vol_avg5 > 0:
        vol_ratio = vol_now / vol_avg5
        details["vol_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 2:
            score += 0.1

    # ---- 年线位置 ----
    if n >= 250:
        ma250 = df["close"].rolling(250).mean().iloc[-1]
        price_now = df["close"].iloc[-1]
        above_year = price_now > ma250
        details["ma250"] = round(ma250, 2)
        details["above_ma250"] = above_year
        score += 0.15 if above_year else -0.15

    # ---- 涨跌停 ----
    if n >= 2:
        pct_change = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
        details["pct_change"] = round(pct_change, 2)
        if pct_change >= 9.5:
            details["limit_up"] = True
            score += 0.2  # 涨停但只能 HOLD，这里给正向辅助信号
        elif pct_change <= -9.5:
            details["limit_down"] = True
            score -= 0.2

    # 归一化到 [-1, +1]
    score = max(-1.0, min(1.0, score))
    return {"score": round(score, 4), "details": details}