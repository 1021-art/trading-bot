#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IDX500 LONG v2_14C2_3_M30p80_MTM37p29 (C2 next-step candidate)
------------------------------------------------------------------------------------
Goal for this iteration:
- Keep MTM MDD within -40% (mark-to-market drawdown constraint)
- Maximize return within that constraint on your provided dataset range

IMPORTANT NOTES (Honesty / Validation):
- With the current pullback-style LONG framework, I did NOT find any setting that reaches
  50x return while also keeping MTM MDD <= -40% on 2021-01-01 ~ 2025-12-28.
- The best candidate found in the targeted search achieves roughly ~9.66x multiple with
  MTM MDD ~ -35.7% (and realized MDD ~ -33.3%), subject to minor rounding differences.

No-lookahead guardrails:
- Macro regime uses 12h bars and is merged by *close_time* (open_time+12h) using merge_asof backward.
- Micro features use 1h bars merged by *close_time* (open_time+1h) using merge_asof backward.
- 30m strategy bars are produced from 15m with an anchor offset to avoid timestamp drift.

"""

import pandas as pd
import numpy as np

# =========================
# ===== USER SETTINGS =====
# =========================
PATH_15M = "Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv"  # default; override via CLI
PATH_1H  = "BTCUSDT_PERP_1h_OHLCV_20210101_to_now_UTC.csv"  # default; override via CLI

# Costs
FEE_RATE = 0.0005
SLIPPAGE = 0.0002

# Capital
INITIAL_CAPITAL = 10000000.0
RISK_PCT = 0.0728
LEV_CAP = 10.0
MAX_POSITION_VALUE = 3000000000.0

# Macro (12h)
MACRO_EMA_LEN = 200
MACRO_ADX_LEN = 14
MACRO_ADX_TH  = 15

# Micro (1h)
EMA_FAST_LEN = 50
RET1_THR       = -0.008
RET30_PREV_THR = -0.01

# Exits (ATR-based)
STOP_ATR_DIST = 0.9
HARD_STOP_PCT = 0.05
TP1_ATR = 0.9
TP2_ATR = 10.0

# TP2 partial runner (take profit partially at TP2, keep a small runner with trailing)
TP2_PARTIAL_CLOSE = 0.95
# Advanced exits
TP1_TIGHTEN_ATR = 0.25
SOFT_WICK_BUFFER_ATR = 0.25
INTRABAR_SOFTSTOP_AFTER_TP1 = True

USE_TRAIL_AFTER_TP1 = True
TRAIL_ATR = 5.0
# Pyramiding (profit-only, after TP1)
ENABLE_PYRAMID = True
MAX_ADDS = 2
ADD_ATR_STEP = 1.0
ADD_SIZE_MULT = 0.32


# Time / cooldown
MAX_HOLD_BARS_30M = 5760
COOLDOWN_HOURS = 12

# Drawdown brake (sizing-scale)
DD_BRAKE = True
DD_STEP_0 = 0.10   # -10%
DD_STEP_1 = 0.20   # -20%
DD_STEP_2 = 0.29   # -30%
DD_STEP_3 = 0.45   # DD magnitude >= 45% -> deepest brake (implemented in dd_scale)
DD_SCALE_0 = 0.82
DD_SCALE_1 = 0.55
DD_SCALE_2 = 0.4
DD_SCALE_3 = 0.3

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
# ===== DATA LOADING ======
# =========================
def _parse_time(df: pd.DataFrame, candidates: list) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            s = df[col]
            # If numeric seconds, parse as unix seconds
            if np.issubdtype(s.dtype, np.number):
                return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            return pd.to_datetime(s, utc=True, errors="coerce")
    raise ValueError(f"Time column not found. Tried columns: {candidates}")

def load_15m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = _parse_time(df, ["datetime", "open_time", "time"])
    df = df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]].copy()

def resample_30m_from_15m(df15: pd.DataFrame) -> pd.DataFrame:
    # Anchor at 15m offset to align with half-hour boundaries consistently
    df = (df15.set_index("open_time")
          .resample("30min", label="left", closed="left", origin="epoch", offset="15min")
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna().reset_index())
    return df

def load_1h(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = _parse_time(df, ["open_time", "time", "datetime"])
    df = df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]].copy()

# =========================
# ===== FEATURES ==========
# =========================
def build_macro_12h(df1h: pd.DataFrame) -> pd.DataFrame:
    df12 = (df1h.set_index("open_time")
            .resample("12h", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna().reset_index())
    df12["ema200"] = ema(df12["close"], MACRO_EMA_LEN)
    df12["adx14"]  = adx(df12, MACRO_ADX_LEN)
    df12["close_time"] = df12["open_time"] + pd.Timedelta(hours=12)
    # Bull regime for LONG
    df12["bull_on"] = ((df12["close"] > df12["ema200"]) & (df12["adx14"] >= MACRO_ADX_TH)).astype(int)
    return df12[["close_time", "bull_on"]].sort_values("close_time")

def build_micro_1h_features(df1h: pd.DataFrame) -> pd.DataFrame:
    out = df1h.copy()
    out["ema50"] = ema(out["close"], EMA_FAST_LEN)
    out["ret1"]  = out["close"] / out["close"].shift(1) - 1.0
    out["atr14"] = atr(out, 14)
    out["close_time"] = out["open_time"] + pd.Timedelta(hours=1)
    feat1h = out[["close_time", "ema50", "ret1", "atr14", "close"]].rename(columns={"close": "close1h"})
    return feat1h.sort_values("close_time")

def merge_features(df30: pd.DataFrame, macro12: pd.DataFrame, feat1h: pd.DataFrame) -> pd.DataFrame:
    df = df30.sort_values("open_time").reset_index(drop=True)
    df = pd.merge_asof(df, macro12, left_on="open_time", right_on="close_time", direction="backward")
    df = pd.merge_asof(df, feat1h, left_on="open_time", right_on="close_time", direction="backward", suffixes=("", "_1h"))
    df["bull_on"] = df["bull_on"].fillna(0).astype(int)
    df["ret30_prev"] = df["close"].pct_change().shift(1)
    return df

# =========================
# ===== EXECUTION HELPERS ==
# =========================
def buy_fill_px(px: float) -> float:  return px * (1.0 + SLIPPAGE)  # buy worse
def sell_fill_px(px: float) -> float: return px * (1.0 - SLIPPAGE)  # sell worse
def stop_fill(open_px: float, stop_px: float) -> float:
    # LONG stop: if gap below stop, fill at open
    return open_px if open_px < stop_px else stop_px

def dd_scale(cur_eq: float, peak_eq: float) -> float:
    if not DD_BRAKE:
        return 1.0
    dd = cur_eq / max(peak_eq, 1e-12) - 1.0
    if dd > -DD_STEP_0:
        return 1.0
    if dd > -DD_STEP_1:
        return DD_SCALE_0
    if dd > -DD_STEP_2:
        return DD_SCALE_1
    if dd > -DD_STEP_3:
        return DD_SCALE_2
    return DD_SCALE_3


# =========================
# ===== BACKTEST ENGINE ===
# =========================
def backtest_long_only(df: pd.DataFrame) -> dict:
    FEE = float(FEE_RATE)
    HOUR_NS = 3_600_000_000_000
    CD_NS = int(COOLDOWN_HOURS) * HOUR_NS

    t_ns = df["open_time"].values.astype("datetime64[ns]").astype(np.int64)
    o_arr = df["open"].to_numpy(dtype=np.float64)
    h_arr = df["high"].to_numpy(dtype=np.float64)
    l_arr = df["low"].to_numpy(dtype=np.float64)
    c_arr = df["close"].to_numpy(dtype=np.float64)

    bull_arr = df["bull_on"].to_numpy(dtype=np.int8)
    ema50_arr = df["ema50"].to_numpy(dtype=np.float64)
    ret1_arr = df["ret1"].to_numpy(dtype=np.float64)
    atr14_arr = df["atr14"].to_numpy(dtype=np.float64)
    c1h_arr = df["close1h"].to_numpy(dtype=np.float64)
    r30p_arr = df["ret30_prev"].to_numpy(dtype=np.float64)

    # State
    eq = float(INITIAL_CAPITAL)
    # --- trade logging (for router backtest stitching) ---
    trade_log = []
    cur_entry_t_ns = None
    cur_eq_before = None
    cur_fees_before = None

    peak = float(INITIAL_CAPITAL)
    realized_mdd = 0.0
    mtm_peak = float(INITIAL_CAPITAL)
    mtm_mdd = 0.0

    fees_paid = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    qty = 0.0
    entry = 0.0
    soft_stop_px = np.nan
    hard_stop_px = np.nan
    tp1 = np.nan
    tp2 = np.nan
    trade_pnl_acc = 0.0
    hold = 0
    partial = False
    pending_exit = False
    atr_entry = np.nan
    block_until_ns = np.int64(-9_223_372_036_854_775_808)

    # Trail state (confirmed by previous close)
    hh_close = np.nan
    # Pyramid state
    adds_done = 0
    tp2_done = False
    next_add_px = np.nan


    # Counters
    cnt_entry = 0
    cnt_hard = 0
    cnt_soft = 0
    cnt_soft_intra = 0
    cnt_soft_close = 0
    cnt_tp1 = 0
    cnt_tp2 = 0
    cnt_time = 0
    cnt_guard = 0  # Zombie/invalid-state forced exits

    def close_all(fill_px: float, reason: int):
        nonlocal eq, fees_paid, trade_pnl_acc, qty, trades, wins, gross_profit, gross_loss
        nonlocal trade_log, cur_entry_t_ns, cur_eq_before, cur_fees_before
        nonlocal entry, soft_stop_px, hard_stop_px, tp1, tp2, hold, partial, pending_exit, atr_entry, hh_close, adds_done, next_add_px
        nonlocal cnt_hard, cnt_soft, cnt_tp2, cnt_time, cnt_soft_intra, cnt_soft_close, cnt_guard

        if qty == 0.0:
            return

        fill = sell_fill_px(fill_px)
        fee = abs(qty) * fill * FEE
        pnl = qty * (fill - entry) - fee

        eq += pnl
        fees_paid += fee
        trade_pnl_acc += pnl

        trades += 1
        # trade log: finalize trade
        if (cur_entry_t_ns is not None) and (cur_eq_before is not None):
            eq_after = float(eq)
            eq_before = float(cur_eq_before)
            trade_fees = float(fees_paid - (cur_fees_before if cur_fees_before is not None else 0.0))
            trade_log.append({
                'entry_time': pd.to_datetime(cur_entry_t_ns),
                'exit_time': pd.to_datetime(int(t_ns[i])),
                'eq_before': eq_before,
                'eq_after': eq_after,
                'ret_pct': (eq_after / eq_before - 1.0) if eq_before > 0 else 0.0,
                'trade_pnl': float(trade_pnl_acc),
                'trade_fees': float(trade_fees),
                'reason': int(reason),
            })
        cur_entry_t_ns = None
        cur_eq_before = None
        cur_fees_before = None
        if trade_pnl_acc > 0.0:
            wins += 1
            gross_profit += trade_pnl_acc
        else:
            gross_loss += -trade_pnl_acc

        # reason: 1=Hard, 2=Soft(Intra), 5=Soft(Close), 3=TP2, 4=Time
        if reason == 1:
            cnt_hard += 1
        elif reason == 2:
            cnt_soft += 1
            cnt_soft_intra += 1
        elif reason == 5:
            cnt_soft += 1
            cnt_soft_close += 1
        elif reason == 3:
            cnt_tp2 += 1
        elif reason == 4:
            cnt_time += 1
        elif reason == 6:
            cnt_guard += 1

        qty = 0.0
        entry = 0.0
        soft_stop_px = np.nan
        hard_stop_px = np.nan
        tp1 = np.nan
        tp2 = np.nan
        trade_pnl_acc = 0.0
        hold = 0
        partial = False
        pending_exit = False
        adds_done = 0
        next_add_px = np.nan
        tp2_done = False
        atr_entry = np.nan
        hh_close = np.nan

    def take_tp1():
        nonlocal eq, fees_paid, trade_pnl_acc, qty, partial, soft_stop_px, cnt_tp1, atr_entry, next_add_px, adds_done
        if qty == 0.0 or partial:
            return

        fill = sell_fill_px(tp1)
        qty_c = qty * 0.5
        fee = abs(qty_c) * fill * FEE
        pnl = qty_c * (fill - entry) - fee

        eq += pnl
        fees_paid += fee
        trade_pnl_acc += pnl

        qty -= qty_c
        partial = True
        cnt_tp1 += 1

        # Initialize pyramid ladder after TP1
        if ENABLE_PYRAMID and (adds_done == 0):
            next_add_px = float(entry + (TP1_ATR + ADD_ATR_STEP) * float(atr_entry))

        # Tighten stop upward (towards entry)
        if np.isfinite(atr_entry):
            ss = float(entry - TP1_TIGHTEN_ATR * float(atr_entry))
            soft_stop_px = max(float(soft_stop_px), ss)

    def take_tp2_partial():
        nonlocal eq, fees_paid, trade_pnl_acc, qty, cnt_tp2, tp2_done, adds_done, next_add_px, soft_stop_px, atr_entry
        if qty == 0.0 or tp2_done:
            return

        # Close a fraction at TP2, keep a runner protected by trailing/soft stop
        close_frac = float(TP2_PARTIAL_CLOSE)
        close_frac = min(max(close_frac, 0.0), 0.95)
        qty_c = qty * close_frac

        fill = sell_fill_px(tp2)
        fee = abs(qty_c) * fill * FEE
        pnl = qty_c * (fill - entry) - fee

        eq += pnl
        fees_paid += fee
        trade_pnl_acc += pnl

        qty -= qty_c
        tp2_done = True
        cnt_tp2 += 1

        # Disable further adds after TP2 partial
        adds_done = MAX_ADDS
        next_add_px = np.nan

        # Extra protection: tighten stop closer after TP2
        if np.isfinite(atr_entry):
            ss2 = float(entry - 0.0 * float(atr_entry))
            soft_stop_px = max(float(soft_stop_px), ss2)

    n = len(o_arr)
    for i in range(n):
        t_i = t_ns[i]
        o = o_arr[i]; h = h_arr[i]; l = l_arr[i]; c = c_arr[i]

        open_px = o  # alias for clarity in gap-safe fills
        bull = bull_arr[i]
        ema50 = ema50_arr[i]
        ret1 = ret1_arr[i]
        atr14 = atr14_arr[i]
        c1h = c1h_arr[i]
        r30p = r30p_arr[i]

        # MTM (mark-to-market) at OPEN
        if qty != 0.0:
            mtm_eq = eq + qty * (o - entry)
        else:
            mtm_eq = eq
        if mtm_eq > mtm_peak:
            mtm_peak = mtm_eq
        dd = mtm_eq / mtm_peak - 1.0
        if dd < mtm_mdd:
            mtm_mdd = dd

        exited_this_bar = False

        # -----------------------------
        # 1) Manage existing position
        # -----------------------------
        if qty != 0.0:
            # Zombie/invalid-state guard: if any critical level is NaN/inf, force-exit at OPEN
            if (not np.isfinite(entry)) or (entry <= 0) or (not np.isfinite(soft_stop_px)) or (not np.isfinite(hard_stop_px)) or (not np.isfinite(tp1)) or (not np.isfinite(tp2)):
                close_all(o, 6)
                exited_this_bar = True
            if not exited_this_bar:
                hold += 1

            # (Trail reference update) use prev close only to avoid intrabar lookahead
            if i > 0:
                prev_close = c_arr[i-1]
                if np.isnan(hh_close):
                    hh_close = float(prev_close)
                else:
                    hh_close = max(float(hh_close), float(prev_close))

            # A) Pending Soft Stop (Close-based) -> Exit at OPEN
            if pending_exit:
                close_all(o, 5)
                exited_this_bar = True

            # B) Intrabar checks
            if (not exited_this_bar) and (qty != 0.0):
                # Time Exit
                if hold >= MAX_HOLD_BARS_30M:
                    close_all(o, 4)
                    exited_this_bar = True

                if (not exited_this_bar) and (qty != 0.0):
                    # Hard Stop
                    if l <= hard_stop_px:
                        px = stop_fill(o, hard_stop_px)
                        close_all(px, 1)
                        exited_this_bar = True

                    else:
                        # Trailing (after TP1)
                        if USE_TRAIL_AFTER_TP1 and partial and np.isfinite(atr_entry) and np.isfinite(hh_close):
                            trail_px = float(hh_close) - TRAIL_ATR * float(atr_entry)
                            soft_stop_px = max(float(soft_stop_px), trail_px)

                        # Intrabar Soft Stop (Pre-TP1 w/ Buffer)
                        buf = (SOFT_WICK_BUFFER_ATR * atr_entry) if np.isfinite(atr_entry) else 0.0
                        allow_intra = ((not partial) or INTRABAR_SOFTSTOP_AFTER_TP1)
                        if allow_intra and (l <= (soft_stop_px - buf)):
                            px = stop_fill(o, soft_stop_px)
                            close_all(px, 2)
                            exited_this_bar = True

                        else:
                            # TP1 then TP2
                            if (not partial) and (h >= tp1):
                                take_tp1()

                            # TP2 first (conservative). If TP2 hits, do not add on same bar.
                            if (qty != 0.0) and (h >= tp2):
                                take_tp2_partial()
                            else:
                                # Pyramiding (profit-only, after TP1) - intrabar trigger
                                if (ENABLE_PYRAMID and partial and (not exited_this_bar) and (qty != 0.0) and
                                    np.isfinite(next_add_px) and (adds_done < MAX_ADDS) and (h >= next_add_px)):
                                    fill = buy_fill_px(max(open_px, next_add_px))
                                    qty_add = qty * ADD_SIZE_MULT
                                    # cap by leverage/position value
                                    cap = min(eq * LEV_CAP, MAX_POSITION_VALUE)
                                    cur_val = qty * fill
                                    add_val = qty_add * fill
                                    if (cur_val + add_val) > cap:
                                        qty_add = max((cap - cur_val) / fill, 0.0)
                                    if qty_add > 0.0:
                                        fee = abs(qty_add) * fill * FEE
                                        eq -= fee
                                        fees_paid += fee
                                        trade_pnl_acc -= fee
                                        # average entry update
                                        entry = (entry * qty + fill * qty_add) / (qty + qty_add)
                                        qty += qty_add
                                        adds_done += 1
                                        next_add_px = float(next_add_px + ADD_ATR_STEP * float(atr_entry))


        # -----------------------------
        # 2) Entry
        # -----------------------------
        entered_this_bar = False
        if (qty == 0.0) and (not exited_this_bar) and (t_i >= block_until_ns):
            if (bull == 1 and np.isfinite(ema50) and np.isfinite(ret1) and
                np.isfinite(atr14) and np.isfinite(c1h) and np.isfinite(r30p)):

                # Pullback entry in bull regime
                if (c1h > ema50) and (ret1 <= RET1_THR) and (r30p <= RET30_PREV_THR):
                    scale = dd_scale(eq, peak)

                    entry = buy_fill_px(o)
                    atr_entry = float(atr14)

                    soft_stop_px = entry - STOP_ATR_DIST * atr_entry
                    hard_stop_px = entry * (1.0 - HARD_STOP_PCT)
                    tp1 = entry + TP1_ATR * atr_entry
                    tp2 = entry + TP2_ATR * atr_entry

                    stop_dist = max(entry - soft_stop_px, 1e-9)
                    qty = (eq * RISK_PCT * scale) / stop_dist

                    cap = min(eq * LEV_CAP, MAX_POSITION_VALUE)
                    if qty * entry > cap:
                        qty = cap / entry

                    fee = abs(qty) * entry * FEE
                    eq -= fee
                    fees_paid += fee
                    trade_pnl_acc = -fee
                    # trade log: mark entry
                    cur_entry_t_ns = int(t_ns[i])
                    cur_eq_before = float(eq + fee)  # equity before paying entry fee
                    cur_fees_before = float(fees_paid - fee)

                    hold = 0
                    partial = False
                    pending_exit = False
                    hh_close = np.nan
                    adds_done = 0
                    tp2_done = False
                    next_add_px = (entry + (TP1_ATR + ADD_ATR_STEP) * atr_entry) if ENABLE_PYRAMID else np.nan
                    entered_this_bar = True
                    cnt_entry += 1

        # -----------------------------
        # 3) Entry-Bar Intrabar
        # -----------------------------
        if entered_this_bar and (qty != 0.0):
            if l <= hard_stop_px:
                px = stop_fill(o, hard_stop_px)
                close_all(px, 1)
                exited_this_bar = True
            else:
                buf = (SOFT_WICK_BUFFER_ATR * atr_entry) if np.isfinite(atr_entry) else 0.0
                if l <= (soft_stop_px - buf):
                    px = stop_fill(o, soft_stop_px)
                    close_all(px, 2)
                    exited_this_bar = True
                else:
                    if h >= tp1:
                        take_tp1()

                    # TP2 first (conservative)
                    if (qty != 0.0) and (h >= tp2):
                        take_tp2_partial()
                    else:
                        # Pyramiding (profit-only, after TP1) - intrabar trigger
                        if (ENABLE_PYRAMID and partial and (not exited_this_bar) and (qty != 0.0) and
                            np.isfinite(next_add_px) and (adds_done < MAX_ADDS) and (h >= next_add_px)):
                            fill = buy_fill_px(max(open_px, next_add_px))
                            qty_add = qty * ADD_SIZE_MULT
                            cap = min(eq * LEV_CAP, MAX_POSITION_VALUE)
                            cur_val = qty * fill
                            add_val = qty_add * fill
                            if (cur_val + add_val) > cap:
                                qty_add = max((cap - cur_val) / fill, 0.0)
                            if qty_add > 0.0:
                                fee = abs(qty_add) * fill * FEE
                                eq -= fee
                                fees_paid += fee
                                trade_pnl_acc -= fee
                                entry = (entry * qty + fill * qty_add) / (qty + qty_add)
                                qty += qty_add
                                adds_done += 1
                                next_add_px = float(next_add_px + ADD_ATR_STEP * float(atr_entry))


        # -----------------------------
        # 4) Soft Stop Trigger (Close-Based)
        # -----------------------------
        if (qty != 0.0) and (not exited_this_bar) and (not pending_exit) and partial:
            if INTRABAR_SOFTSTOP_AFTER_TP1:
                buf = (SOFT_WICK_BUFFER_ATR * atr_entry) if np.isfinite(atr_entry) else 0.0
                if c <= (soft_stop_px - buf):
                    pending_exit = True
            else:
                if c <= soft_stop_px:
                    pending_exit = True

        # -----------------------------
        # 5) Cooldown
        # -----------------------------
        if exited_this_bar:
            block_until_ns = (t_i // HOUR_NS) * HOUR_NS + CD_NS

        # Realized DD tracking
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < realized_mdd:
            realized_mdd = dd

        # Safety assert (optional, but keep it cheap)
        if not np.isfinite(eq):
            raise RuntimeError("Equity became non-finite (NaN/inf). Check sizing/fees.")

    # Final Close at last close (conservative)
    if qty != 0.0:
        close_all(float(c_arr[-1]), 4)

    multiple = eq / float(INITIAL_CAPITAL)
    winrate = (wins / max(trades, 1)) * 100.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    net_profit = eq - float(INITIAL_CAPITAL)
    fees_over_net = (fees_paid / net_profit) * 100.0 if net_profit > 0 else float("nan")

    return {
        "Multiple": float(multiple),
        "Realized_MDD_pct": float(realized_mdd * 100.0),
        "MTM_MDD_pct": float(mtm_mdd * 100.0),
        "Trades": int(trades),
        "WinRate_pct": float(winrate),
        "PF": float(pf),
        "Fees_over_NetProfit_pct": float(fees_over_net),
        "trade_log": trade_log,

        "Counters": {
            "entries": int(cnt_entry),
            "hard_stop_exits": int(cnt_hard),
            "soft_stop_exits": int(cnt_soft),
            "soft_stop_intra": int(cnt_soft_intra),
            "soft_stop_close": int(cnt_soft_close),
            "tp1_exits": int(cnt_tp1),
            "tp2_exits": int(cnt_tp2),
            "time_exits": int(cnt_time),
            "guard_exits": int(cnt_guard),
        }
    }

def main(args):
    print("Loading 15m...")
    df15 = load_15m(args.path_15m)
    print("Resampling to 30m...")
    df30 = resample_30m_from_15m(df15)
    print("Loading 1h...")
    df1h = load_1h(args.path_1h)

    print("Building macro 12h...")
    macro12 = build_macro_12h(df1h)
    print("Building micro 1h...")
    feat1h = build_micro_1h_features(df1h)

    print("Merging features...")
    df = merge_features(df30, macro12, feat1h)

    print("Backtesting baseline (C2)...")
    global TP2_PARTIAL_CLOSE, DD_SCALE_0, DD_SCALE_1, DD_STEP_2
    _saved = (TP2_PARTIAL_CLOSE, DD_SCALE_0, DD_SCALE_1, DD_STEP_2)

    # Baseline C2 settings (selected baseline)
    TP2_PARTIAL_CLOSE = 0.92
    DD_SCALE_0 = 0.85
    DD_SCALE_1 = 0.60
    DD_STEP_2 = 0.30
    res_base = backtest_long_only(df)

    # Restore candidate defaults (this file)
    TP2_PARTIAL_CLOSE, DD_SCALE_0, DD_SCALE_1, DD_STEP_2 = _saved

    print("Backtesting candidate (C2.3)...")
    res = backtest_long_only(df)


    def _print_res(tag: str, r: dict):
        print("")
        print(f"==== RESULTS ({tag}) ====")
        print(f"Multiple: {r['Multiple']:.4f}x")
        print(f"Realized MDD: {r['Realized_MDD_pct']:.2f}%")
        print(f"MTM MDD: {r['MTM_MDD_pct']:.2f}%")
        print(f"Trades: {r['Trades']}")
        print(f"WinRate: {r['WinRate_pct']:.2f}%")
        print(f"PF: {r['PF']:.3f}")
        print(f"Fees/NetProfit(%): {r['Fees_over_NetProfit_pct']:.2f}%")
        c = r["Counters"]
        print("Counters:")
        print(f"  Entries     : {c['entries']}")
        print(f"  Hard Stops  : {c['hard_stop_exits']}")
        print(f"  Soft Stops  : {c['soft_stop_exits']} (Intrabar: {c['soft_stop_intra']}, Close: {c['soft_stop_close']})")
        print(f"  TP1 Exits   : {c['tp1_exits']}")
        print(f"  TP2 Exits   : {c['tp2_exits']}")
        print(f"  Time Exits  : {c['time_exits']}")
        print(f"  Guard Exits : {c['guard_exits']}")

    def _print_delta(new: dict, old: dict):
        print("")
        print("---- Delta vs Baseline (C2) ----")
        print(f"Multiple Δ: {new['Multiple']-old['Multiple']:+.4f}x")
        print(f"Realized MDD Δ: {new['Realized_MDD_pct']-old['Realized_MDD_pct']:+.2f}%p")
        print(f"MTM MDD Δ: {new['MTM_MDD_pct']-old['MTM_MDD_pct']:+.2f}%p")
        print(f"Trades Δ: {new['Trades']-old['Trades']:+d}")
        print(f"WinRate Δ: {new['WinRate_pct']-old['WinRate_pct']:+.2f}%p")
        print(f"PF Δ: {new['PF']-old['PF']:+.3f}")
        print(f"Fees/NetProfit Δ: {new['Fees_over_NetProfit_pct']-old['Fees_over_NetProfit_pct']:+.2f}%p")

    _print_res("Baseline C2", res_base)
    _print_res("Candidate C2.3", res)
    _print_delta(res, res_base)


    return res
if __name__ == "__main__":
    import argparse
    import os
    import json
    import pandas as pd

    ap = argparse.ArgumentParser(description="IDX500 engine standalone runner (optional). When imported, use backtest_long_only() directly.")
    ap.add_argument("--path_15m", default=PATH_15M, help="15m CSV path")
    ap.add_argument("--path_1h",  default=PATH_1H,  help="1h CSV path")
    ap.add_argument("--out_trades", default="idx_used_trades.csv", help="output trade log csv")
    ap.add_argument("--out_summary", default="idx_summary.json", help="output summary json")
    args = ap.parse_args()

    if not os.path.exists(args.path_15m):
        raise FileNotFoundError(f"15m CSV not found: {args.path_15m}")
    if not os.path.exists(args.path_1h):
        raise FileNotFoundError(f"1h CSV not found: {args.path_1h}")

    # Run
    res = main(args)

    # Save outputs (optional)
    try:
        tl = res.get("trade_log", [])
        if isinstance(tl, list) and len(tl) > 0:
            pd.DataFrame(tl).to_csv(args.out_trades, index=False)
    except Exception as e:
        print(f"[WARN] failed to write trade log: {e}")

    try:
        out = {k: v for k, v in res.items() if k != "trade_log"}
        with open(args.out_summary, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write summary: {e}")

