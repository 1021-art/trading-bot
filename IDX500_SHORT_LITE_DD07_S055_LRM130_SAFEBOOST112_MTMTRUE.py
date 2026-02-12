#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IDX500 SHORT-LITE v3 (aux mildbear short, PF-tuned)
------------------------------------------------------------------------------------
* v50 FIX:
  - Fixes NameError on soft stop counters by explicitly defining:
      cnt_soft_intra, cnt_soft_close
  - Fixes time parsing to correctly handle epoch seconds/ms/us/ns (robust).
  - Fixes policy bug: Close-based soft stop must NOT trigger Pre-TP1.
    (Pre-TP1 soft stop is intrabar-only with wick buffer.)
  - TP1 이후 정책:
      INTRABAR_SOFTSTOP_AFTER_TP1=True  -> TP1 이후에도 intrabar soft stop 허용(윅버퍼 유지)
      INTRABAR_SOFTSTOP_AFTER_TP1=False -> TP1 이후 Close-only(패닉홀드)만 허용

Design Logic (Policy):
1) Pre-TP1: Soft Stop triggers Intrabar ONLY if High breaches (SoftStop + WickBuffer).
2) Post-TP1:
   - If INTRABAR_SOFTSTOP_AFTER_TP1=True: Intrabar Soft Stop allowed (wick-buffered).
   - If False: Close-based Soft Stop only (Panic Hold, next bar open exit).
3) Hard Stop: Always active Intrabar (Emergency).
4) Order-of-events (same bar): HardStop -> SoftStop -> TP.
5) No lookahead: merge_asof backward using feature close_time <= 30m open_time.

* TUNING (2022+ optimization, trade-count unchanged):
  - HARD_STOP_PCT: 0.055 -> 0.045
  - STOP_ATR_DIST: 2.0   -> 1.8
  - Goal: higher multiple while keeping realized MDD suppressed.
"""
# =========================================================
# LOCKED CONFIG: Candidate 1 (WinRate>=71%, RealizedMDD<=25.5%)
# - MAX_ADDS      = 5
# - ADD_ATR_STEP  = 0.75
# - ADD_SIZE_MULT = 0.45
# =========================================================


import pandas as pd
import numpy as np

# =========================
# ===== USER SETTINGS =====
# =========================
PATH_15M = r"/mnt/data/Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv"
PATH_1H  = r"/mnt/data/BTCUSDT_PERP_1h_OHLCV_20210101_to_now_UTC.csv"

# Costs
FEE_RATE = 0.0005
SLIPPAGE = 0.0002
DEV_START = "2021-01-01"  # include 2021+ for evaluation

# Capital / Risk
INITIAL_CAPITAL = 10_000_000.0

# Risk Ramp (early equity protection): scale down risk until equity grows
RISK_RAMP = True
RAMP_LEVELS = [(1.05, 0.6), (1.15, 0.9)]  # (eq_multiple_threshold, risk_mult) [A_SAFE]

# Risk-on (late compounding) - only when equity already > 2.0x and macro conditions are favorable
RISK_ON_AFTER_MULT = 1.4
RISK_ON_MULT = 1.30
RISK_ON_ATRP_MAX = 0.018
RISK_ON_MACRO_ADX_MIN = 20.0

RISK_PCT = 0.1
LITE_RISK_MULT = 1.30  # <--- added: global multiplier for LITE sizing
ALLOW_BEAR0 = True
BEAR0_ADX_MIN = 22.0
BEAR0_ATRP_MAX = 0.025
BEAR0_RET1_THR = -0.017
BEAR0_RISK_MULT = 0.30  # sizing multiplier when bear_on==0

LEV_CAP = 20.0
MAX_POSITION_VALUE = 3_000_000_000.0

# Macro (12H)
MACRO_EMA_LEN = 200
MACRO_ADX_LEN = 14
MACRO_ADX_TH  = 22
MACRO_ATRP_TH = 0.05
MACRO_ADX_LO_TH = 15

# Micro (1H)
EMA_FAST_LEN = 50
RET1_THR = -0.012
RET30_PREV_THR = -0.008
ADX_ENTRY_THR = 22.0
ATRP_ENTRY_MAX = 0.022

# Exits
STOP_ATR_DIST = 1.4
HARD_STOP_PCT = 0.040
TP1_ATR = 1.0
TP2_ATR = 6.0

# ====== [FIXED PARAMS: Best PF @ Multiple>=7.6x] ======
TP1_TIGHTEN_ATR = 0.00
SOFT_WICK_BUFFER_ATR = 0.35
INTRABAR_SOFTSTOP_AFTER_TP1 = True
# ======================================================

MAX_HOLD_BARS_30M = 144

# =========================================================
# Pyramiding (ported from LONG engine) - profit-only adds after TP1
# =========================================================
ENABLE_PYRAMID = False
MAX_ADDS = 0
ADD_ATR_STEP = 0.90
ADD_SIZE_MULT = 0.35

# =========================================================
# Drawdown brake (ported from LONG engine) - deep-only safety
# - Does NOT trigger in current backtest (Realized MDD ~ -29%)
# - Only reduces sizing if realized DD exceeds 30%+
# =========================================================
DD_BRAKE = True
DD_STEP_0 = 0.07   # -7%
DD_STEP_1 = 9.99   # disabled
DD_STEP_2 = 9.99   # disabled
DD_STEP_3 = 9.99   # disabled
DD_SCALE_0 = 0.55
DD_SCALE_1 = 1.00
DD_SCALE_2 = 1.00
DD_SCALE_3 = 1.00

COOLDOWN_HOURS = 3


# ===== Adaptive Rolling Entry Cap (No-lookahead) =====
TRADECAP_WINDOW_DAYS = 7
CAP_WEAK   = 2
CAP_MID    = 4
CAP_STRONG = 6
CAP_ADX_MID_TH = 22.0
CAP_ADX_STRONG_TH = 35.0
CAP_ATRP_MID_TH = 0.050
CAP_ATRP_STRONG_TH = 0.060

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
def _to_utc_datetime(s: pd.Series) -> pd.Series:
    """
    Robust UTC datetime parser:
    - numeric epoch seconds/ms/us/ns auto-detected by magnitude
    - ISO strings supported
    """
    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors="coerce")
        med = np.nanmedian(x.values)
        if np.isnan(med):
            return pd.to_datetime(x, utc=True, errors="coerce")
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def _parse_time(df: pd.DataFrame, candidates: list) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return _to_utc_datetime(df[col])
    raise ValueError(f"Time column not found: {candidates} in columns={df.columns.tolist()}")

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df

def load_15m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_cols(df)
    df["open_time"] = _parse_time(df, ["datetime", "open_time", "time"])
    df = df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]].copy()

def resample_30m_from_15m(df15: pd.DataFrame) -> pd.DataFrame:
    # Keep same anchoring behavior as prior versions
    df = (df15.set_index("open_time")
          .resample("30min", label="left", closed="left", origin="epoch", offset="15min")
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna().reset_index())
    return df

def load_1h(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_cols(df)
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
    df12["atr14"]  = atr(df12, 14)
    df12["atrp"]   = (df12["atr14"] / df12["close"].replace(0, np.nan)).fillna(0.0)
    df12["close_time"] = df12["open_time"] + pd.Timedelta(hours=12)
    df12["ema200_down"] = (df12["ema200"] < df12["ema200"].shift(12)).astype(int)
    df12["bear_on"] = ((df12["close"] < df12["ema200"]) & (df12["ema200_down"] == 1) & ((df12["adx14"] >= MACRO_ADX_TH) | ((df12["adx14"] >= MACRO_ADX_LO_TH) & (df12["atrp"] >= MACRO_ATRP_TH)))).astype(int)
    return df12[["close_time", "bear_on", "adx14", "atrp"]].sort_values("close_time")

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
    # No lookahead: only bars whose close_time <= open_time
    df = pd.merge_asof(df, macro12, left_on="open_time", right_on="close_time", direction="backward")
    df = pd.merge_asof(df, feat1h, left_on="open_time", right_on="close_time", direction="backward", suffixes=("", "_1h"))
    df["bear_on"] = df["bear_on"].fillna(0).astype(int)
    df["adx14"] = df["adx14"].fillna(0.0).astype(float)
    df["atrp"]  = df["atrp"].fillna(0.0).astype(float)
    df["ret30_prev"] = df["close"].pct_change().shift(1)
    return df

# =========================
# ===== EXECUTION HELPERS ==
# =========================
def sell_fill_px(px: float) -> float: return px * (1.0 - SLIPPAGE)
def buy_fill_px(px: float) -> float:  return px * (1.0 + SLIPPAGE)

def risk_ramp(eq: float, macro_adx: float = 0.0, macro_atrp: float = 0.0) -> float:
    """Equity-based risk ramp (no lookahead).
    - Early: conservative (per RAMP_LEVELS)
    - Late: risk-on ONLY if equity already compounded and macro conditions are favorable
    """
    if not RISK_RAMP:
        return 1.0
    m = eq / INITIAL_CAPITAL
    # Late compounding boost (only in favorable macro)
    if (m >= RISK_ON_AFTER_MULT) and (macro_adx >= RISK_ON_MACRO_ADX_MIN) and (macro_atrp <= RISK_ON_ATRP_MAX):
        return RISK_ON_MULT
    # Early protection ramp
    for thr, mult in RAMP_LEVELS:
        if m < thr:
            return mult
    return 1.0



def dd_scale(eq: float, peak: float) -> float:
    """Return sizing scale based on realized drawdown (deep-only brake)."""
    if peak <= 0.0:
        return 1.0
    dd = max(0.0, (peak - eq) / peak)
    if dd >= DD_STEP_3:
        return DD_SCALE_3
    if dd >= DD_STEP_2:
        return DD_SCALE_2
    if dd >= DD_STEP_1:
        return DD_SCALE_1
    if dd >= DD_STEP_0:
        return DD_SCALE_0
    return 1.0

def stop_fill(open_px: float, stop_px: float) -> float:
    # Gap-through stop handling: if open is worse (above stop for short), fill at open
    return open_px if open_px > stop_px else stop_px

# =========================
# ===== BACKTEST ENGINE ===
# =========================
def backtest_short_only(df: pd.DataFrame) -> dict:
    FEE = float(FEE_RATE)
    HOUR_NS = 3_600_000_000_000
    CD_NS = int(COOLDOWN_HOURS) * HOUR_NS
    HARDSTOP_COOLDOWN_HOURS = 24  # 하드스탑 이후 추가 쿨다운(시간)
    HARDSTOP_CD_NS = np.int64(HARDSTOP_COOLDOWN_HOURS) * np.int64(HOUR_NS)

    t_ns = df["open_time"].values.astype("datetime64[ns]").astype(np.int64)
    o_arr = df["open"].to_numpy(dtype=np.float64)
    h_arr = df["high"].to_numpy(dtype=np.float64)
    l_arr = df["low"].to_numpy(dtype=np.float64)
    c_arr = df["close"].to_numpy(dtype=np.float64)

    bear_arr = df["bear_on"].to_numpy(dtype=np.int8)
    macro_adx_arr = df["adx14"].to_numpy(dtype=np.float64)
    macro_atrp_arr = df["atrp"].to_numpy(dtype=np.float64)
    ema50_arr = df["ema50"].to_numpy(dtype=np.float64)
    ret1_arr = df["ret1"].to_numpy(dtype=np.float64)
    atr14_arr = df["atr14"].to_numpy(dtype=np.float64)
    c1h_arr = df["close1h"].to_numpy(dtype=np.float64)
    r30p_arr = df["ret30_prev"].to_numpy(dtype=np.float64)

    # Equity / DD
    eq = float(INITIAL_CAPITAL)
    peak = float(INITIAL_CAPITAL)
    realized_mdd = 0.0
    mtm_peak = float(INITIAL_CAPITAL)
    mtm_mdd = 0.0
    eq_curve = np.empty(len(df), dtype=np.float64)  # patched

    # Stats
    fees_paid = 0.0
    trade_log = []
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    trades_log = []  # patched: per-trade exit log

    # Position state
    qty = 0.0
    entry_time = pd.NaT
    eq_before_entry = 0.0
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
    adds_done = 0
    next_add_px = np.nan
    block_until_ns = np.int64(-9_223_372_036_854_775_808)

    # Rolling cap state
    window_ns = np.int64(TRADECAP_WINDOW_DAYS) * np.int64(24) * np.int64(HOUR_NS)
    entry_times = []  # list of t_i


    # Counters
    cnt_entry = 0
    cnt_hard = 0
    cnt_soft = 0
    cnt_soft_intra = 0
    cnt_soft_close = 0
    cnt_tp1 = 0
    cnt_tp2 = 0
    cnt_time = 0

    def close_all(fill_px: float, reason: int, exit_time):
        nonlocal eq, fees_paid, trade_pnl_acc, qty, trades, wins, gross_profit, gross_loss
        nonlocal entry, soft_stop_px, hard_stop_px, tp1, tp2, hold, partial, pending_exit, atr_entry, adds_done, next_add_px, block_until_ns
        nonlocal cnt_hard, cnt_soft, cnt_tp2, cnt_time, cnt_soft_intra, cnt_soft_close

        if qty == 0.0:
            return

        fill = buy_fill_px(fill_px)
        fee = abs(qty) * fill * FEE
        pnl = qty * (fill - entry) - fee  # qty<0 short
        eq += pnl
        fees_paid += fee
        trade_pnl_acc += pnl

        trades += 1
        # patched: log trade exit (exit time = current bar open_time)
        trades_log.append({
            "exit_time": pd.to_datetime(int(current_t_i), unit='ns', utc=True),
            "pnl": float(trade_pnl_acc),
            "win": int(trade_pnl_acc > 0.0),
            "reason": int(reason),
        })
        if trade_pnl_acc > 0.0:
            wins += 1
            gross_profit += trade_pnl_acc
        else:
            gross_loss += -trade_pnl_acc

        
        # trade log (return uses equity before entry; includes fees/partials accumulated into eq)
        try:
            ret_pct = (eq / eq_before_entry - 1.0) if eq_before_entry > 0 else 0.0
        except Exception:
            ret_pct = 0.0
        trade_log.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "ret_pct": float(ret_pct),
            "reason": int(reason),
        })

# reason: 1 hard, 2 soft(intra), 3 tp2, 4 time, 5 soft(close)
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

        # 하드스탑 후 재진입 억제 (기존 1H 쿨다운 + 추가 쿨다운)
        if reason == 1:
            block_until_ns = max(block_until_ns, current_t_i + HARDSTOP_CD_NS)

        # reset
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
        adds_done = 0
        next_add_px = np.nan

    def take_tp1():
        nonlocal eq, fees_paid, trade_pnl_acc, qty, partial, soft_stop_px, cnt_tp1, atr_entry, adds_done, next_add_px
        if qty == 0.0 or partial:
            return

        fill = buy_fill_px(tp1)
        qty_c = qty * 0.5
        fee = abs(qty_c) * fill * FEE
        pnl = qty_c * (fill - entry) - fee
        eq += pnl
        fees_paid += fee
        trade_pnl_acc += pnl

        qty -= qty_c
        partial = True

        # Initialize pyramid ladder after TP1
        if ENABLE_PYRAMID and (adds_done == 0) and np.isfinite(atr_entry):
            next_add_px = float(entry - (TP1_ATR + ADD_ATR_STEP) * float(atr_entry))
        cnt_tp1 += 1

        # tighten soft stop after TP1
        if np.isfinite(atr_entry):
            ss = float(entry + TP1_TIGHTEN_ATR * float(atr_entry))
            soft_stop_px = min(float(soft_stop_px), ss)

    current_t_i = np.int64(0)  # patched
    n = len(o_arr)
    for i in range(n):
        t_i = t_ns[i]
        current_t_i = t_i  # patched
        o = o_arr[i]; h = h_arr[i]; l = l_arr[i]; c = c_arr[i]
        df_time = df['open_time'].iloc[i]

        bear = bear_arr[i]
        ema50 = ema50_arr[i]
        ret1 = ret1_arr[i]
        atr14 = atr14_arr[i]
        c1h = c1h_arr[i]
        r30p = r30p_arr[i]

        # MTM DD
        mtm_eq = eq + (qty * (h - entry) if qty != 0.0 else 0.0)
        if mtm_eq > mtm_peak:
            mtm_peak = mtm_eq
        dd_mtm = mtm_eq / mtm_peak - 1.0
        if dd_mtm < mtm_mdd:
            mtm_mdd = dd_mtm

        exited_this_bar = False

        # 1) Manage position
        if qty != 0.0:
            hold += 1

            # A) pending close-soft-stop -> exit at OPEN
            if pending_exit:
                close_all(o, 5, df_time)
                exited_this_bar = True

            if (not exited_this_bar) and (qty != 0.0):
                # time exit
                if hold >= MAX_HOLD_BARS_30M:
                    close_all(o, 4, df_time)
                    exited_this_bar = True

                if (not exited_this_bar) and (qty != 0.0):
                    # Hard stop
                    if h >= hard_stop_px:
                        px = stop_fill(o, hard_stop_px)
                        close_all(px, 1, df_time)
                        exited_this_bar = True

                    # Intrabar soft stop (wick-buffered)
                    elif (((not partial) or INTRABAR_SOFTSTOP_AFTER_TP1) and
                          (h >= (soft_stop_px + (SOFT_WICK_BUFFER_ATR * atr_entry if np.isfinite(atr_entry) else 0.0)))):
                        px = stop_fill(o, soft_stop_px)
                        close_all(px, 2, df_time)
                        exited_this_bar = True

                    else:
                        # TP logic
                        if (not partial) and (l <= tp1):
                            take_tp1()
                        if (qty != 0.0) and (l <= tp2):
                            close_all(tp2, 3, df_time)
                            exited_this_bar = True
                        else:
                            # Pyramiding (profit-only, after TP1) - intrabar trigger
                            if (ENABLE_PYRAMID and partial and (not exited_this_bar) and (qty != 0.0) and
                                np.isfinite(next_add_px) and (adds_done < MAX_ADDS) and (l <= next_add_px)):
                                fill = sell_fill_px(min(o, next_add_px))
                                qty_add = qty * ADD_SIZE_MULT  # qty is negative for short
                                cap = min(eq * LEV_CAP, MAX_POSITION_VALUE)
                                cur_val = abs(qty) * fill
                                add_val = abs(qty_add) * fill
                                if (cur_val + add_val) > cap:
                                    max_add = max((cap - cur_val) / fill, 0.0)
                                    qty_add = -max_add if qty_add < 0 else max_add
                                if abs(qty_add) > 0.0:
                                    fee = abs(qty_add) * fill * FEE
                                    eq -= fee
                                    fees_paid += fee
                                    trade_pnl_acc -= fee
                                    entry = (entry * qty + fill * qty_add) / (qty + qty_add)
                                    qty += qty_add
                                    adds_done += 1
                                    next_add_px = float(next_add_px - ADD_ATR_STEP * float(atr_entry))

        # 2) Entry
        entered_this_bar = False
        if (qty == 0.0) and (not exited_this_bar) and (t_i >= block_until_ns):
            # Adaptive rolling entry cap (No-lookahead)
            madx = float(macro_adx_arr[i])
            matrp = float(macro_atrp_arr[i])
            if (madx >= CAP_ADX_STRONG_TH) and (matrp >= CAP_ATRP_STRONG_TH):
                capN = int(CAP_STRONG)
            elif (madx >= CAP_ADX_MID_TH) and (matrp >= CAP_ATRP_MID_TH):
                capN = int(CAP_MID)
            else:
                capN = int(CAP_WEAK)

            # drop old entries
            while entry_times and (t_i - entry_times[0]) > window_ns:
                entry_times.pop(0)
            if len(entry_times) >= capN:
                pass
            
            elif (((bear == 1) or (ALLOW_BEAR0 and (madx >= BEAR0_ADX_MIN) and (matrp <= BEAR0_ATRP_MAX))) and np.isfinite(ema50) and np.isfinite(ret1) and
                np.isfinite(atr14) and np.isfinite(c1h) and np.isfinite(r30p)):
                if (c1h < ema50) and (madx >= ADX_ENTRY_THR) and (matrp <= ATRP_ENTRY_MAX) and ( ((bear == 1) and ((ret1 <= RET1_THR) or (r30p <= RET30_PREV_THR))) or ((bear != 1) and (ret1 <= BEAR0_RET1_THR)) ):
                    entry = sell_fill_px(o)
                    entry_time = df_time
                    eq_before_entry = eq
                    soft_stop_px = entry + STOP_ATR_DIST * float(atr14)
                    hard_stop_px = entry * (1.0 + HARD_STOP_PCT)
                    tp1 = entry - TP1_ATR * float(atr14)
                    tp2 = entry - TP2_ATR * float(atr14)

                    atr_entry = float(atr14)  # capture at entry

                    stop_dist = max(soft_stop_px - entry, 1e-9)
                    scale = dd_scale(eq, peak) if DD_BRAKE else 1.0
                    ramp = risk_ramp(eq, madx, matrp)

                    # --- Safe-regime boost (MTM-friendly) ---
                    # Goal: increase multiple with minimal MTM impact by ONLY boosting in low-vol + strong-trend regime,
                    # and scaling down in higher-vol / weak-trend regimes.
                    boost = 1.0
                    # Low volatility + strong macro trend (safe to add a bit)
                    if (matrp <= 0.016) and (madx >= 28.0):
                        boost = 1.12
                    # High volatility or weak trend (protect MTM)
                    if (matrp >= 0.022) or (madx <= 18.0):
                        boost = min(boost, 0.85)
                    qty = -(eq * RISK_PCT * scale * ramp * (1.0 if bear == 1 else BEAR0_RISK_MULT) * LITE_RISK_MULT) / stop_dist
                    qty *= boost

                    cap = min(eq * LEV_CAP, MAX_POSITION_VALUE)
                    if abs(qty) * entry > cap:
                        qty = -cap / entry

                    fee = abs(qty) * entry * FEE
                    eq -= fee
                    fees_paid += fee
                    trade_pnl_acc = -fee

                    hold = 0
                    partial = False
                    pending_exit = False
                    adds_done = 0
                    next_add_px = np.nan
                    entered_this_bar = True
                    cnt_entry += 1
                    entry_times.append(t_i)


        # 3) Entry-bar intrabar processing
        if entered_this_bar and (qty != 0.0):
            if h >= hard_stop_px:
                px = stop_fill(o, hard_stop_px)
                close_all(px, 1, df_time)
                exited_this_bar = True

            elif (((not partial) or INTRABAR_SOFTSTOP_AFTER_TP1) and
                  (h >= (soft_stop_px + (SOFT_WICK_BUFFER_ATR * atr_entry if np.isfinite(atr_entry) else 0.0)))):
                px = stop_fill(o, soft_stop_px)
                close_all(px, 2, df_time)
                exited_this_bar = True

            else:
                if (not partial) and (l <= tp1):
                    take_tp1()
                if (qty != 0.0) and (l <= tp2):
                    close_all(tp2, 3, df_time)
                    exited_this_bar = True


        # 3.5) Close-based soft stop (PRE-TP1, BUFFERED) [Option 3]
        # - 목적: '슬금슬금' 올라 종가가 soft stop 근처에 붙는 상황을 방치하는 것을 완화하되,
        #         노이즈성 미세 돌파는 윅버퍼 기준으로 필터링
        # - 주의: 본 조건은 (soft_stop + buffer) 기준이라 intrabar soft stop과 대부분 중복될 수 있음.
        if (qty != 0.0) and (not exited_this_bar) and (not pending_exit) and (not partial):
            buf = (SOFT_WICK_BUFFER_ATR * atr_entry) if np.isfinite(atr_entry) else 0.0
            if c >= (soft_stop_px + buf):
                pending_exit = True

        # 4) Close-based soft stop (POST-TP1 ONLY)
        if (qty != 0.0) and (not exited_this_bar) and (not pending_exit) and partial:
            if INTRABAR_SOFTSTOP_AFTER_TP1:
                # buffer concept 유지: close도 버퍼 넘었을 때만 패닉홀드 시작
                buf = (SOFT_WICK_BUFFER_ATR * atr_entry) if np.isfinite(atr_entry) else 0.0
                if c >= (soft_stop_px + buf):
                    pending_exit = True
            else:
                if c >= soft_stop_px:
                    pending_exit = True

        # 5) Cooldown (no re-entry within 1H bucket)
        if exited_this_bar:
            block_until_ns = max(block_until_ns, (t_i // HOUR_NS) * HOUR_NS + CD_NS)

        # Realized DD
        if eq > peak:
            peak = eq
        dd_real = eq / peak - 1.0
        if dd_real < realized_mdd:
            realized_mdd = dd_real
        eq_curve[i] = eq

    # Final close (use time exit reason 4 to keep counters consistent)
    if qty != 0.0:
        close_all(float(c_arr[-1]), 4, df_time)

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
        "trades_log": trades_log,
        "equity_curve": pd.DataFrame({"open_time": df["open_time"].values, "equity": eq_curve}),
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
        }
    }

def main():
    print("Loading 15m...")
    df15 = load_15m(PATH_15M)
    # Filter to 2022+ (UTC) for evaluation
    df15 = df15[df15["open_time"] >= pd.Timestamp(DEV_START, tz="UTC")].reset_index(drop=True)
    print("Resampling to 30m...")
    df30 = resample_30m_from_15m(df15)
    print("Loading 1h...")
    df1h = load_1h(PATH_1H)
    df1h = df1h[df1h["open_time"] >= pd.Timestamp(DEV_START, tz="UTC")].reset_index(drop=True)

    print("Building macro 12h...")
    macro12 = build_macro_12h(df1h)
    print("Building micro 1h...")
    feat1h = build_micro_1h_features(df1h)

    print("Merging features (no lookahead)...")
    df = merge_features(df30, macro12, feat1h)

    print("Backtesting (V52)...")
    res = backtest_short_only(df)

    print(f"\n==== RESULTS (V52: RISK_PCT={RISK_PCT:.3f}, HARD_STOP_PCT={HARD_STOP_PCT:.3f}) ====")
    print(f"Multiple: {res['Multiple']:.4f}x")
    print(f"Realized MDD: {res['Realized_MDD_pct']:.2f}%")
    print(f"MTM MDD: {res['MTM_MDD_pct']:.2f}%")
    print(f"Trades: {res['Trades']}")
    print(f"WinRate: {res['WinRate_pct']:.2f}%")
    print(f"PF: {res['PF']:.3f}")
    print(f"Fees/NetProfit(%): {res['Fees_over_NetProfit_pct']:.2f}%")
    c = res["Counters"]
    print("Counters:")
    print(f"  Entries     : {c['entries']}")
    print(f"  Hard Stops  : {c['hard_stop_exits']}")
    print(f"  Soft Stops  : {c['soft_stop_exits']} (Intrabar: {c['soft_stop_intra']}, Close: {c['soft_stop_close']})")
    print(f"  TP1 Exits   : {c['tp1_exits']}")
    print(f"  TP2 Exits   : {c['tp2_exits']}")
    print(f"  Time Exits  : {c['time_exits']}")


    # ===== Patched: Quarterly summary for 2021 =====
    eq_df = res.get("equity_curve")
    tl = res.get("trades_log", [])
    if isinstance(eq_df, pd.DataFrame) and len(eq_df) > 0:
        d = eq_df.copy()
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        d = d[(d["open_time"] >= pd.Timestamp("2021-01-01", tz="UTC")) & (d["open_time"] < pd.Timestamp("2022-01-01", tz="UTC"))]
        if len(d) > 0:
            d["Quarter"] = d["open_time"].dt.to_period("Q")
            qeq = d.groupby("Quarter").agg(StartEq=("equity","first"), EndEq=("equity","last"))
            qeq["Return%"] = (qeq["EndEq"]/qeq["StartEq"] - 1.0) * 100.0

            tdf = pd.DataFrame(tl)
            if len(tdf) > 0:
                tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
                tdf = tdf[(tdf["exit_time"] >= pd.Timestamp("2021-01-01", tz="UTC")) & (tdf["exit_time"] < pd.Timestamp("2022-01-01", tz="UTC"))]
                if len(tdf) > 0:
                    tdf["Quarter"] = tdf["exit_time"].dt.to_period("Q")
                    qt = tdf.groupby("Quarter").agg(Trades=("pnl","count"), WinRate=("win","mean"))
                    qt["WinRate%"] = qt["WinRate"] * 100.0
                    qt = qt.drop(columns=["WinRate"])
                else:
                    qt = pd.DataFrame(columns=["Trades","WinRate%"])
            else:
                qt = pd.DataFrame(columns=["Trades","WinRate%"])

            q = qeq.join(qt, how="left").fillna({"Trades":0, "WinRate%":0.0})
            print("\n==== 2021 Quarterly (Return / Trades / WinRate) ====")
            for idx, row in q.iterrows():
                print(f"{idx}: Return={row['Return%']:+.2f}%  Trades={int(row['Trades'])}  WinRate={row['WinRate%']:.2f}%  StartEq={row['StartEq']:.0f}  EndEq={row['EndEq']:.0f}")

if __name__ == "__main__":
    main()

# =========================
# ===== TRADELOG API ======
# =========================
def get_tradelog():
    """Run full backtest and return trade log list[dict]."""
    df15 = load_15m(PATH_15M)
    df15 = df15[df15["open_time"] >= pd.Timestamp(DEV_START, tz="UTC")].reset_index(drop=True)
    df30 = resample_30m_from_15m(df15)
    df1h = load_1h(PATH_1H)
    df1h = df1h[df1h["open_time"] >= pd.Timestamp(DEV_START, tz="UTC")].reset_index(drop=True)
    macro12 = build_macro_12h(df1h)
    feat1h = build_micro_1h_features(df1h)
    df = merge_features(df30, macro12, feat1h)
    res = backtest_short_only(df)
    return res["trade_log"]