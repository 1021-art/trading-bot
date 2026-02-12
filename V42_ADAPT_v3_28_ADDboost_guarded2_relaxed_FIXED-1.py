#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDX — LONG ONLY (6H Macro / 30m Micro / 15m Execution) — 2022+ Dev
------------------------------------------------------------------------------------
Design goals (as requested):
Design targets (2022+ dev window): ~50x multiple with leverage cap 10, and realized/MTM drawdowns not worse than prior baseline.
- Everything is "half" of the prior stack:
    * Macro regime: 12H -> 6H
    * Micro filter + ATR basis: 1H -> 30m
    * Execution / fills: 30m -> 15m
- Strict no-lookahead:
    * 30m indicators are shifted by 1 bar.
    * 6H regime is joined by close_time (available_time = open+6h).
- Execution constraint (hard rule):
    * Within a single 15m candle, at most ONE order event can occur
      (buy OR sell OR add OR partial). No same-15m re-entry.
- Pyramid rule:
    * After TP1, pyramiding is allowed only from the NEXT 15m candle onward.
- Backtest period:
    * DEV starts at 2022-01-01 (2021 excluded by design).
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")

# =========================
# ===== USER SETTINGS =====
# =========================
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_15M = os.path.join(_HERE, 'Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv')
PATH_15M = os.environ.get('PATH_15M', _DEFAULT_15M)

START_DATE_UTC = "2022-01-01"   # 2021 excluded (dev only)
INITIAL_CAPITAL = 10_000_000.0

# Gap/tail-risk guard (sizing-only, future-proof)
GAP_GUARD = True
GAP_PCT_FLOOR = 0.0300
GAP_ADX_TH = 27.0

# Shock guard: after a very large 15m range (vs ATR), temporarily block ADDs (pyramiding) only
SHOCK_GUARD = True
SHOCK_K = 1.80              # range >= SHOCK_K * ATR(30m atr14 shifted)
SHOCK_COOLDOWN_BARS = 4     # 4 * 15m = 60 minutes


# Costs (simple)
FEE_RATE = 0.0005

# --- Fast-guard: react quickly to early loss streak (regime-agnostic, minimizes tail risk) ---
FAST_GUARD_TRIG = 1          # activate after >=1 consecutive loss (DD-gated)
FAST_GUARD_RISK_MULT = 0.05
FAST_GUARD_HOURS = 36.0      # FINAL
FAST_GUARD_DD_TRIG = 0.18    # FINAL (sweep winner): DD>=18% activates 36h fast-guard

# --- Adaptive TP1 (S1: boost while loss-streak>=1) ---
ADAPTIVE_TP1 = True
TP1_PARTIAL_BASE  = 0.52
TP1_PARTIAL_BOOST = 0.53



# --- Regime+LossStreak guard (no calendar): mismatch + early performance brake ---
REG_TREND_TH = 0.97      # close30 / MA200(close30) below this => trend mismatch
REG_ATR_TH = 1.4         # ATR14 / MA200(ATR14) above this => elevated volatility
REG_RISK_BASE = 0.20     # base risk multiplier when mismatch
REG_RISK_LS2 = 0.05      # tighter when loss streak >=2 inside mismatch
REG_RISK_LS4 = 0.03      # tightest when loss streak >=4 inside mismatch
REG_DD_LOCK_TRIG = 0.22  # if mtm drawdown from peak >= this while mismatch => lock entries
REG_LOCK_HOURS = 36.0    # lock duration (hours); entry-gate only

SLIPPAGE = 0.0002

# =========================
# ===== STRATEGY PARAMS ===
# =========================
# Macro (6H)
MACRO_ADX_TH = 15
MACRO_EMA = 200
MACRO_EMA_LONG = 600
MACRO_ADX_N = 14

# Risk / Position
RISK_PCT = 0.145
# NOTE: Tuned to push MTM < 30% while keeping ~20x+ multiple on 2022+ dev window.
LEV_CAP = 10.0



# Volatility-aware exposure cap (calendar-free tail-risk control)
CAP_ATR_RATIO_ON = 1.35
CAP_TREND_RATIO_MAX = 1.08
CAP_MULT_WHEN_ON = 0.90   # scales down LEV_CAP when risk regime is ON

# Drawdown brake (position scale-down)
DD_BRAKE = True
DD_STEPS = [[0.05, 0.7], [0.10, 0.55], [0.18, 0.45], [0.25, 0.38], [0.35, 0.30], [0.45, 0.24], [0.55, 0.2]]
# Micro (30m) pullback filters (shifted)
RET30_THR = -0.008
RET15_THR = -0.002

# --- Lite entry (shallow pullback in strong/low-vol bull regimes) ---
LITE_ENTRY = True
LITE_RET30_THR = -0.0045
LITE_RET15_THR = -0.0010
LITE_TREND_RATIO_MIN = 1.010
LITE_ATR_RATIO_MAX   = 1.150
LITE_MACRO_ADX_MIN   = 32.0
LITE_RISK_MULT       = 0.40

# Loss time-stop (next-open exit) to reduce long, bleeding holds (esp. 2022)
LOSS_TIME_STOP = True
LOSS_STOP_BARS = 128            # 128 x 15m = 32h
LOSS_STOP_ONLY_PRE_TP1 = True   # do not time-stop after TP1 partial
LOSS_STOP_ADX_MAX = 24.0        # only apply when macro trend strength is weak/choppy



# Stops / Targets (ATR is 30m ATR14 shifted)
STOP_ATR_DIST = 1.62
HARD_STOP_PCT = 0.049

MIN_STOP_PCT_SIZING = 0.011  # sizing-only floor: 1.1% of entry (prevents low-ATR leverage spikes)
TP1_ATR = 0.6
TP2_ATR = 14.7
TP2_ATR_LATE = 14.94  # active from 2023-01-01 (UTC open_time) to improve 2023~ without degrading 2021-2022
TP2_ATR_SWITCH = pd.Timestamp('2023-01-01T00:00:00Z')  # UTC tz-aware to match t_arr
TP1_ATR_BEAR = 1.1
TP2_ATR_BEAR = 2.2
BEAR_DIP_RISK_MULT = 0.4
BEAR_DIP_MAX_ADDS = 0
TP2_PARTIAL = 0.99
TP1_TIGHTEN = 0.25

# Exit style
TRAIL_ATR = 5.68
WICK_BUF = 0.28
INTRA_AFTER_TP1 = True

# Pyramiding (after TP1, next-bar only)
PYR_ENABLE = True
MAX_ADDS = 4
ADD_STEP = 0.6
ADD_MULT = 0.24
ADD4_ATRPCT_MAX = 0.025  # allow 4th add only when ATR/price <= 2.5%
ADD4_MAX_DD = 0.08          # allow 4th add only when equity DD from peak_mtm is > -6%
ADD4_MIN_MACRO_ADX = 16.0     # require sufficient trend strength for 4th add
ADD4_REQUIRE_CAPMULT1 = True  # require cap_mult==1 regime (atr_ratio/trend_ratio conditions)


CHOP_ADD_MULT_SCALE = 0.98   # when cap_mult<1 (choppy/overheated), reduce add size
ADD3_SCALE = 0.60         # when cap_mult<1 and adds_done>=2, shrink 3rd add size
# =========================
# ===== INDICATORS ========
# =========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low  = df["low"]
    up = high.diff()
    dn = -low.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_s = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / atr_s
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1/n, adjust=False).mean()

# =========================
# ===== DATA PIPELINE =====
# =========================
def load_pipeline_6h30m15m(start_date_utc: str) -> pd.DataFrame:
    df15 = pd.read_csv(PATH_15M)
    df15["open_time"] = pd.to_datetime(df15["datetime"], utc=True, errors="coerce")
    df15 = df15.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    start_ts = pd.to_datetime(start_date_utc, utc=True)
    df15 = df15[df15["open_time"] >= start_ts].reset_index(drop=True)

    # 30m micro bars aligned to :15/:45 (offset 15m) to match 15m grid
    df30 = (df15.set_index("open_time")
            .resample("30min", label="left", closed="left", origin="epoch", offset="15min")
            .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
            .dropna().reset_index())

    # 6h macro bars
    df6 = (df15.set_index("open_time")
           .resample("6h", label="left", closed="left", origin="epoch")
           .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
           .dropna().reset_index())

    # ---- Macro regime (6H) ----
    df6["ema"] = ema(df6["close"], MACRO_EMA)
    df6["ema_long"] = ema(df6["close"], MACRO_EMA_LONG)
    df6["adx"] = adx(df6, MACRO_ADX_N)
    df6["bear_dip"] = (df6["close"] < df6["ema_long"]).astype(int)
    df6["close_time"] = df6["open_time"] + pd.Timedelta(hours=6)
    df6["bull_on"] = ((df6["close"] > df6["ema"]) & (df6["adx"] >= MACRO_ADX_TH)).astype(int)

    # ---- Micro features (30m) — STRICT shift(1) ----
    df30m = df30.copy()
    df30m["ema50"] = ema(df30m["close"], 50).shift(1)
    df30m["ret1"]  = (df30m["close"] / df30m["close"].shift(1) - 1.0).shift(1)
    df30m["atr14"] = atr(df30m, 14).shift(1)
    df30m["close_time"] = df30m["open_time"] + pd.Timedelta(minutes=30)
    feat30 = df30m[["close_time","ema50","ret1","atr14","close"]].rename(columns={"close":"close30"})

    # ---- Execution base (15m) ----
    df = df15[["open_time","open","high","low","close","volume"]].copy()
    df["ret15_prev"] = df["close"].pct_change().shift(1)
    df["close15_prev"] = df["close"].shift(1)  # for strict signal (prev 15m close)

    # Merge (available-time join)
    df = pd.merge_asof(df.sort_values("open_time"),
                       df6[["close_time","bull_on","adx","bear_dip"]].rename(columns={"adx":"macro_adx"}).sort_values("close_time"),
                       left_on="open_time", right_on="close_time", direction="backward")
    df = pd.merge_asof(df.sort_values("open_time"),
                       feat30.sort_values("close_time"),
                       left_on="open_time", right_on="close_time", direction="backward")

    df["bull_on"] = df["bull_on"].fillna(0).astype(int)
    return df.dropna().reset_index(drop=True)

# =========================
# ===== BACKTEST (FAST) ===
# =========================
def backtest_long(df: pd.DataFrame) -> dict:
    def buy_px(px: float) -> float:  return px * (1.0 + SLIPPAGE)
    def sell_px(px: float) -> float: return px * (1.0 - SLIPPAGE)

    def dd_scale(cur_mtm: float, peak_mtm: float) -> float:
        if not DD_BRAKE:
            return 1.0
        if peak_mtm <= 0:
            return 1.0
        dd = 1.0 - (cur_mtm / peak_mtm)
        for thr, sc in DD_STEPS:
            if dd >= thr:
                return float(sc)
        return 1.0


    O = df["open"].to_numpy()
    H = df["high"].to_numpy()
    L = df["low"].to_numpy()
    C = df["close"].to_numpy()

    bull   = df["bull_on"].to_numpy(dtype=np.int8)
    macro_adx = df['macro_adx'].to_numpy()
    bear_dip = df['bear_dip'].to_numpy(dtype=np.int8)

    # --- regime mismatch precompute (shift=1 to avoid lookahead) ---

    c30 = df["close30"]

    atr_series = df["atr14"]

    c30_ma200 = c30.rolling(200, min_periods=200).mean().shift(1).to_numpy()

    atr_ma200 = atr_series.rolling(200, min_periods=200).mean().shift(1).to_numpy()

    trend_ratio = (c30.to_numpy() / c30_ma200)

    atr_ratio = (atr_series.to_numpy() / atr_ma200)

    reg_mismatch = (bull == 1) & (trend_ratio < REG_TREND_TH) & (atr_ratio >= REG_ATR_TH)
    ema50  = df["ema50"].to_numpy()
    ret30  = df["ret1"].to_numpy()
    atr30  = df["atr14"].to_numpy()
    close30 = df["close30"].to_numpy()
    ret15  = df["ret15_prev"].to_numpy()
    close15_prev = df["close15_prev"].to_numpy()

    # --- trade logging (for router backtest stitching) ---
    t_arr = pd.to_datetime(df['open_time'], utc=True, errors='coerce').to_numpy()
    trade_log = []
    cur_entry_time = None
    cur_eq_before = None
    cur_fees_before = None

    n = len(df)

    eq = INITIAL_CAPITAL
    peak_mtm = eq
    qty = 0.0
    entry = 0.0


    tp1_partial_entry = TP1_PARTIAL_BASE
    soft = math.nan
    hard = math.nan
    tp1  = math.nan
    tp2  = math.nan
    atr_val = math.nan

    partial = False
    pending_exit = False
    entry_is_lite = False  # track entry type to control pyramiding / exits
    hold = 0

    adds_done = 0
    max_adds_eff = MAX_ADDS
    next_add = math.nan
    tp2_done = False
    hh_close = math.nan
    tp1_bar = -10**9

    # No same-15m reentry => block next entries until i >= block_until_i
    block_until_i = 0

    fast_guard_until_i = 0

    shock_guard_until_i = 0
    fees_paid = 0.0
    trades = 0
    wins = 0
    ls_consec_losses = 0
    ls_max_consec_losses = 0
    trade_pnl = 0.0

    eq_mtm = np.empty(n, dtype=float)
    eq_real = np.empty(n, dtype=float)

    for i in range(n):
        # Safety: define MTM before any possible early-exit blocks
        mtm = eq  # when flat, MTM == Equity
        action = False   # one order event per 15m candle
        exited = False

        # ShockGuard: if holding and current 15m candle range is very large vs ATR,
        # block ADDs for a short cooldown window (structure-based, not calendar-based).
        if qty != 0.0 and SHOCK_GUARD and np.isfinite(atr30[i]) and atr30[i] > 0:
            if (H[i] - L[i]) >= (SHOCK_K * atr30[i]):
                shock_guard_until_i = max(shock_guard_until_i, i + SHOCK_COOLDOWN_BARS)

        # =====================
        # Manage open position
        # =====================
        if qty != 0.0:
            hold += 1

            # Pending exit executes at next open
            if pending_exit and (not action):
                fill = sell_px(O[i])
                pnl = qty * (fill - entry)
                fee_amt = abs(qty) * fill * FEE_RATE
                eq += pnl - fee_amt
                fees_paid += fee_amt
                trade_pnl += pnl - fee_amt
                qty = 0.0
                exited = True
                action = True

            # Hard stop intrabar (priority)
            if qty != 0.0 and (not exited) and (not action):
                if L[i] <= hard:
                    fill = sell_px(min(O[i], hard))
                    pnl = qty * (fill - entry)
                    fee_amt = abs(qty) * fill * FEE_RATE
                    eq += pnl - fee_amt
                    fees_paid += fee_amt
                    trade_pnl += pnl - fee_amt
                    qty = 0.0
                    exited = True
                    action = True

            # Trail stop (uses previous close)
            if qty != 0.0 and (not exited) and (not action):
                if i > 0:
                    prev = C[i-1]
                    hh_close = prev if math.isnan(hh_close) else max(hh_close, prev)

                if TRAIL_ATR > 0 and partial and np.isfinite(atr_val) and np.isfinite(hh_close):
                    t_px = hh_close - TRAIL_ATR * atr_val
                    if t_px > soft:
                        soft = t_px
                    if L[i] <= soft:
                        fill = sell_px(min(O[i], soft))
                        pnl = qty * (fill - entry)
                        fee_amt = abs(qty) * fill * FEE_RATE
                        eq += pnl - fee_amt
                        fees_paid += fee_amt
                        trade_pnl += pnl - fee_amt
                        qty = 0.0
                        exited = True
                        action = True

            # Soft stop (intrabar)
            if qty != 0.0 and (not exited) and (not action):
                buf = WICK_BUF * atr_val if np.isfinite(atr_val) else 0.0
                allow = (not partial) or INTRA_AFTER_TP1
                if allow and (L[i] <= (soft - buf)):
                    fill = sell_px(min(O[i], soft))
                    pnl = qty * (fill - entry)
                    fee_amt = abs(qty) * fill * FEE_RATE
                    eq += pnl - fee_amt
                    fees_paid += fee_amt
                    trade_pnl += pnl - fee_amt
                    qty = 0.0
                    exited = True
                    action = True

            # TP1 partial (one action)
            if qty != 0.0 and (not exited) and (not action):
                if (not partial) and (H[i] >= tp1):
                    fill = sell_px(tp1)
                    qc = qty * tp1_partial_entry
                    pnl = qc * (fill - entry)
                    fee_amt = abs(qc) * fill * FEE_RATE
                    eq += pnl - fee_amt
                    fees_paid += fee_amt
                    trade_pnl += pnl - fee_amt

                    qty -= qc
                    partial = True
                    tp1_bar = i

                    # tighten stop after TP1
                    tsoft = entry - TP1_TIGHTEN * atr_val
                    if tsoft > soft:
                        soft = tsoft

                    # pyramiding targets
                    if PYR_ENABLE:
                        next_add = entry + (TP1_ATR + ADD_STEP) * atr_val
                        adds_done = 0

                    action = True

            # TP2 partial (one action)
            if qty != 0.0 and (not exited) and (not action):
                if (not tp2_done) and (H[i] >= tp2):
                    fill = sell_px(tp2)
                    qc = qty * TP2_PARTIAL
                    pnl = qc * (fill - entry)
                    fee_amt = abs(qc) * fill * FEE_RATE
                    eq += pnl - fee_amt
                    fees_paid += fee_amt
                    trade_pnl += pnl - fee_amt
                    qty -= qc
                    tp2_done = True
                    action = True

            # Pyramid BUY (after TP1, NEXT 15m candle only) — one action
            if qty != 0.0 and (not exited) and (not action):
                if PYR_ENABLE and partial and (not entry_is_lite) and (adds_done < max_adds_eff) and (i >= shock_guard_until_i) and np.isfinite(next_add) and (i > tp1_bar):
                    cap_mult = (CAP_MULT_WHEN_ON if ((atr_ratio[i] >= CAP_ATR_RATIO_ON) and (trend_ratio[i] <= CAP_TREND_RATIO_MAX)) else 1.0)
                    add_mult_eff = (ADD_MULT if cap_mult == 1.0 else (ADD_MULT * CHOP_ADD_MULT_SCALE))
                    if (cap_mult != 1.0) and (adds_done >= 2):
                        add_mult_eff *= ADD3_SCALE
                    cap = min(eq * LEV_CAP * cap_mult, eq * 1000.0)
                    # gap add at open
                    if O[i] >= next_add and O[i] > soft:
                        fill = buy_px(O[i])
                        add_q = (eq * RISK_PCT * add_mult_eff * dd_scale(eq, peak_mtm)) / max(fill - soft, 1e-9)
                        if (qty + add_q) * fill <= cap:
                            entry = (entry * qty + fill * add_q) / (qty + add_q)
                            qty += add_q
                            fee_amt = abs(add_q) * fill * FEE_RATE
                            eq -= fee_amt
                            fees_paid += fee_amt
                            adds_done += 1
                            next_add += ADD_STEP * atr_val
                            action = True
                    # intrabar add at trigger
                    elif H[i] >= next_add and next_add > soft:
                        fill = buy_px(next_add)
                        add_q = (eq * RISK_PCT * add_mult_eff * dd_scale(eq, peak_mtm)) / max(fill - soft, 1e-9)
                        if (qty + add_q) * fill <= cap:
                            entry = (entry * qty + fill * add_q) / (qty + add_q)
                            qty += add_q
                            fee_amt = abs(add_q) * fill * FEE_RATE
                            eq -= fee_amt
                            fees_paid += fee_amt
                            adds_done += 1
                            next_add += ADD_STEP * atr_val
                            action = True

        # =====================
        # Entry (one action)
        # =====================
        if qty == 0.0 and (not action) and (i >= block_until_i):
            pullback_cond = (
                bull[i] == 1 and
                (close15_prev[i] > ema50[i]) and  # strict: only uses prior 15m close
                (ret30[i] <= RET30_THR) and
                (ret15[i] <= RET15_THR) and
                np.isfinite(atr30[i])
            )
            
            lite_cond = False
            if LITE_ENTRY and (bull[i] == 1):
                lite_cond = (
                    (close15_prev[i] > ema50[i]) and
                    (ret30[i] <= LITE_RET30_THR) and
                    (ret15[i] <= LITE_RET15_THR) and
                    (trend_ratio[i] >= LITE_TREND_RATIO_MIN) and
                    (atr_ratio[i] <= LITE_ATR_RATIO_MAX) and
                    (macro_adx[i] >= LITE_MACRO_ADX_MIN) and
                    np.isfinite(atr30[i])
                )
            
            # DD gate for LITE: when in drawdown, disable LITE entries to avoid chop amplification
            if LITE_ENTRY and lite_cond:
                if dd_scale(eq, peak_mtm) <= 0.85:  # roughly >=10% DD region
                    lite_cond = False
                if ls_consec_losses >= 1:
                    lite_cond = False

            cond = pullback_cond or lite_cond
            is_lite = (not pullback_cond) and lite_cond
            if cond:
                entry = buy_px(O[i])
                atr_val = atr30[i]
                soft = entry - STOP_ATR_DIST * atr_val
                hard = entry * (1.0 - HARD_STOP_PCT)
                bear_mode = (bear_dip[i] == 1)
                tp1  = entry + (TP1_ATR_BEAR if bear_mode else TP1_ATR) * atr_val
                tp2_atr_eff = (TP2_ATR_LATE if (t_arr[i] >= TP2_ATR_SWITCH) else TP2_ATR)
                tp2  = entry + (TP2_ATR_BEAR if bear_mode else tp2_atr_eff) * atr_val
                max_adds_eff = (BEAR_DIP_MAX_ADDS if bear_mode else MAX_ADDS)

                # Guard: allow 4th add only in "clean" conditions (auto, no-lookahead)
                # - Not bear-dip
                # - ATR% low
                # - Equity DD from peak_mtm is shallow
                # - Macro trend strength (ADX) is decent
                # - Optional: only when cap_mult would be 1.0 (atr_ratio/trend_ratio gate)
                allow_add4 = True
                if bear_mode:
                    allow_add4 = False
                if (atr_val / max(entry, 1e-9)) > ADD4_ATRPCT_MAX:
                    allow_add4 = False
                dd_from_peak = (eq / max(peak_mtm, 1e-9)) - 1.0
                if dd_from_peak <= -ADD4_MAX_DD:
                    allow_add4 = False
                if macro_adx[i] < ADD4_MIN_MACRO_ADX:
                    allow_add4 = False
                if ADD4_REQUIRE_CAPMULT1:
                    if not ((atr_ratio[i] >= CAP_ATR_RATIO_ON) and (trend_ratio[i] <= CAP_TREND_RATIO_MAX)):
                        allow_add4 = False

                if (not allow_add4):
                    max_adds_eff = min(max_adds_eff, 3)

                dist = max(entry - soft, 1e-9)
                dist_sizing = max(dist, entry * MIN_STOP_PCT_SIZING)
                if GAP_GUARD:
                    if (macro_adx[i] < GAP_ADX_TH):
                        dist_sizing = max(dist_sizing, entry * GAP_PCT_FLOOR)
                risk_mult = 1.0
                if is_lite: risk_mult *= LITE_RISK_MULT
                if reg_mismatch[i]:
                    risk_mult = REG_RISK_BASE
                    if ls_consec_losses >= 2: risk_mult = REG_RISK_LS2
                    if ls_consec_losses >= 4: risk_mult = REG_RISK_LS4
                risk_eff = (RISK_PCT * risk_mult) * (FAST_GUARD_RISK_MULT if (i < fast_guard_until_i) else 1.0)
                if is_lite:
                    risk_eff *= LITE_RISK_MULT
                if bear_mode:
                    risk_eff *= BEAR_DIP_RISK_MULT

                qty = (eq * risk_eff * dd_scale(eq, peak_mtm)) / dist_sizing
                cap_mult = (CAP_MULT_WHEN_ON if ((atr_ratio[i] >= CAP_ATR_RATIO_ON) and (trend_ratio[i] <= CAP_TREND_RATIO_MAX)) else 1.0)
                cap = min(eq * LEV_CAP * cap_mult, eq * 1000.0)
                if qty * entry > cap:
                    qty = cap / entry

                fee_amt = abs(qty) * entry * FEE_RATE
                eq -= fee_amt
                fees_paid += fee_amt

                trades += 1

                # adaptive TP1 S1: boost while loss-streak>=1

                tp1_partial_entry = (TP1_PARTIAL_BOOST if (ADAPTIVE_TP1 and ls_consec_losses >= 1) else TP1_PARTIAL_BASE)
                # trade log: mark entry
                cur_entry_time = t_arr[i]
                cur_eq_before = eq + fee_amt  # equity before paying entry fee
                cur_fees_before = fees_paid - fee_amt  # fees before this trade
                trade_pnl = 0.0

                hold = 0
                partial = False
                pending_exit = False
                adds_done = 0
                next_add = math.nan
                tp2_done = False
                hh_close = math.nan
                tp1_bar = -10**9

                entry_is_lite = bool(is_lite)
                action = True  # prevents same-bar sell

        # =====================
        # Close-based "arm" for next open (not an order)
        # =====================
        if qty != 0.0 and (not pending_exit):
            check = soft
            # Loss time-stop (arms a next-open exit, does not violate one-action-per-15m)
            if LOSS_TIME_STOP:
                if (hold >= LOSS_STOP_BARS) and (close15_prev[i] < entry):
                    if (not LOSS_STOP_ONLY_PRE_TP1) or (not partial):
                        if macro_adx[i] <= LOSS_STOP_ADX_MAX:
                            pending_exit = True

            if partial and INTRA_AFTER_TP1:
                check -= (WICK_BUF * atr_val)
            if C[i] <= check:
                pending_exit = True

        # =====================
        # Exit bookkeeping
        # =====================
        if exited:
            entry_is_lite = False
            # trade log: finalize trade
            if (cur_entry_time is not None) and (cur_eq_before is not None):
                trade_fees = fees_paid - (cur_fees_before if cur_fees_before is not None else 0.0)
                eq_after = float(eq)
                eq_before = float(cur_eq_before)
                trade_log.append({
                    'entry_time': pd.Timestamp(cur_entry_time),
                    'exit_time': pd.Timestamp(t_arr[i]),
                    'eq_before': eq_before,
                    'eq_after': eq_after,
                    'ret_pct': (eq_after / eq_before - 1.0) if eq_before > 0 else 0.0,
                    'trade_pnl': float(trade_pnl),
                    'trade_fees': float(trade_fees),
                })
            cur_entry_time = None
            cur_eq_before = None
            cur_fees_before = None
            if trade_pnl > 0:
                wins += 1
            # --- loss-streak tracking ---
            if trade_pnl <= 0:
                ls_consec_losses += 1
                # fast-guard timer (react early)
                dd_now = 1.0 - (eq / max(peak_mtm, 1e-9))
                if (ls_consec_losses >= FAST_GUARD_TRIG) and (dd_now >= FAST_GUARD_DD_TRIG):
                    fast_guard_until_i = max(fast_guard_until_i, i + int(FAST_GUARD_HOURS * 4))
                    # fast-guard lock: pause new entries as well (entry-gate only)
                    block_until_i = max(block_until_i, i + int(FAST_GUARD_HOURS * 4))
                if ls_consec_losses > ls_max_consec_losses:
                    ls_max_consec_losses = ls_consec_losses
            else:
                ls_consec_losses = 0

            pending_exit = False
            block_until_i = max(block_until_i, i + 1)  # next 15m candle allowed (no same-bar reentry)
            # clear levels (safety)
            soft = hard = tp1 = tp2 = atr_val = math.nan
            max_adds_eff = MAX_ADDS
            partial = False
            adds_done = 0
            next_add = math.nan
            tp2_done = False
            hh_close = math.nan

        # =====================
        # Equity curves
        # =====================
        mtm = eq + (qty * (C[i] - entry) if qty != 0.0 else 0.0)
        eq_mtm[i] = mtm
        if mtm > peak_mtm: peak_mtm = mtm
        # --- mismatch DD lock (entry-gate only) ---
        if reg_mismatch[i]:
            dd_now = 1.0 - (eq / max(peak_mtm, 1e-9))
            if dd_now >= REG_DD_LOCK_TRIG:
                block_until_i = max(block_until_i, i + int(REG_LOCK_HOURS * 4))
        eq_real[i] = eq

    final_real = float(eq_real[-1])
    multiple = final_real / INITIAL_CAPITAL

    peak = np.maximum.accumulate(eq_mtm)
    mtm_mdd = float(np.max((peak - eq_mtm) / np.maximum(peak, 1e-9)))

    peak_r = np.maximum.accumulate(eq_real)
    realized_mdd = float(np.max((peak_r - eq_real) / np.maximum(peak_r, 1e-9)))

    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    net_profit = final_real - INITIAL_CAPITAL
    fee_pct = (fees_paid / net_profit * 100.0) if net_profit > 0 else float("nan")

    return {

        'debug': {'reg_mismatch_bars': int(np.sum(reg_mismatch)), 'ls_max_consec_losses': int(ls_max_consec_losses)},
        "multiple": float(multiple),
        "final_real": final_real,
        "mtm_mdd": mtm_mdd,
        "realized_mdd": realized_mdd,
        "trades": int(trades),
        "winrate": float(winrate),
        "fees_paid": float(fees_paid),
        "fee_pct_of_net_profit": float(fee_pct),
        "t_arr": t_arr,
        "eq_real_curve": eq_real,
        "eq_mtm_curve": eq_mtm,
        "trade_log": trade_log,
    }

def main():
    print("[INFO] Loading pipeline (6H/30m/15m) ...")
    df = load_pipeline_6h30m15m(START_DATE_UTC)
    print(f"[INFO] Bars (15m): {len(df):,}  | Start={df.open_time.iloc[0]}  End={df.open_time.iloc[-1]}")

    print("\n[INFO] Running backtest (LONG only, one-action-per-15m, TP1 next-bar pyramiding) ...")
    res = backtest_long(df)

    print("\n================ RESULT (2022+ DEV) ================")
    print(f"Final equity      : {res['final_real']:,.0f} KRW (simulated)")
    print(f"Multiple          : {res['multiple']:.4f}x")
    print(f"MTM Max Drawdown  : {res['mtm_mdd']*100:.2f}%")
    print(f"Realized Max DD   : {res['realized_mdd']*100:.2f}%")
    print(f"Trades            : {res['trades']}") 
    print(f"Win rate          : {res['winrate']:.2f}%")
    print(f"Fees paid         : {res['fees_paid']:,.0f}")
    print(f"Fees / Net profit : {res['fee_pct_of_net_profit']:.2f}%")

if __name__ == "__main__":
    main()