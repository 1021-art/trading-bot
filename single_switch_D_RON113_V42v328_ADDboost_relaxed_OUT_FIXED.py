#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-switch (no stitch) dual-engine portfolio backtest
- Engines: IDX500_CD12_with_tradelog.py, V42_ADAPT_...py
- Enforces: single position, MIN_REENTRY = 15m bar, no-lookahead (fixed tie-break + state-only sizing)
- Adds: Quality-gated auto ON/OFF (Regime/Quality score) + DD-brake sizing
"""

import sys, importlib.util, json
from datetime import timedelta
import pandas as pd
import numpy as np
import os

import argparse

def _resolve(p: str) -> str:
    """Resolve path: if relative, interpret relative to this script."""
    if p is None:
        return p
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), p)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_15m', default='Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv')
    ap.add_argument('--path_1h',  default='BTCUSDT_PERP_1h_OHLCV_20210101_to_now_UTC.csv')
    ap.add_argument('--idx_engine', default='IDX500_CD12_with_tradelog.py')
    ap.add_argument('--v42_engine', default='V42_ADAPT_v3_28_ADDboost_guarded2_relaxed_FIXED.py')
    ap.add_argument('--out_used_trades', default='dual_switch_used_trades_D_RON113_V42v328_ADDboost_relaxed.csv')
    ap.add_argument('--out_summary', default='dual_switch_summary_D_RON113_V42v328_ADDboost_relaxed.json')
    return ap.parse_args()

# =========================
# Execution constraints
# =========================
MIN_REENTRY_MINUTES = 15
# fixed priority for exact same entry_time (no future info)
TIE_PRIORITY = ["V42", "IDX500"]

# =========================
# Portfolio DD-brake policy (v6 family)
# =========================
T1 = -0.052152
T2 = -0.175622
T3 = -0.198491
BOOT_PEAK_MAX = 1.0

V42_S1 = 1.0
V42_S2 = 0.0
V42_S3 = 0.0
V42_S3_BOOT = 0.0

IDX_S1 = 0.8
IDX_S2 = 1.3
IDX_S3 = 0.25
IDX_S3_BOOT = 0.2

# =========================
# Quality gating (Q5)
# =========================
# score <= SCORE_SKIP  -> OFF (skip)
SCORE_SKIP = 0
RISK_ON_MULT = 1.13
DD_RISKON_CAP = -0.1
DD_RISKON_ON  = -0.08   # risk-on enabled only when dd > this   # if dd <= cap, risk-on (>1.0) is disabled
RISK_ON_SCORE_MIN = 3

# NOTE: Score is computed ONLY from features available at entry time (no future returns).
def quality_score(row: pd.Series) -> int:
    sc = 0
    # bull regime confirmation
    if int(row.get("bull_on", 0) or 0) == 1:
        sc += 2
    # price above ema50
    close = row.get("close", np.nan)
    ema50 = row.get("ema50", np.nan)
    if pd.notna(close) and pd.notna(ema50) and close > ema50:
        sc += 1
    # positive 1h return filter
    ret1 = row.get("ret1", np.nan)
    if pd.notna(ret1) and ret1 > 0:
        sc += 1
    if pd.notna(ret1) and ret1 < -0.01:
        sc -= 2
    # V42-specific "bear dip" penalty
    if row.get("engine") == "V42" and int(row.get("bear_dip", 0) or 0) == 1:
        sc -= 2
    return int(sc)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_dd_size(dd: float, peak: float, engine: str) -> float:
    if dd <= T3 and peak <= BOOT_PEAK_MAX:
        return V42_S3_BOOT if engine == "V42" else IDX_S3_BOOT
    if dd <= T3:
        return V42_S3 if engine == "V42" else IDX_S3
    if dd <= T2:
        return V42_S2 if engine == "V42" else IDX_S2
    if dd <= T1:
        return V42_S1 if engine == "V42" else IDX_S1
    return 1.0


def enrich_trade_log(trade_log, engine: str, df_feat: pd.DataFrame) -> pd.DataFrame:
    # df_feat has tz-aware UTC open_time; trades are UTC-naive -> normalize both to UTC-naive
    dff = df_feat.copy()
    dff["open_time_naive"] = pd.to_datetime(dff["open_time"], utc=True, errors="coerce").dt.tz_convert(None)
    dff = dff.sort_values("open_time_naive")

    tr = pd.DataFrame(trade_log).copy()
    tr["engine"] = engine
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True, errors="coerce").dt.tz_convert(None)
    tr["exit_time"]  = pd.to_datetime(tr["exit_time"],  utc=True, errors="coerce").dt.tz_convert(None)
    tr["ret_pct"] = pd.to_numeric(tr["ret_pct"], errors="coerce").fillna(0.0).astype(float)

    # attach entry-time features (asof backward)
    merged = pd.merge_asof(
        tr.sort_values("entry_time"),
        dff,
        left_on="entry_time",
        right_on="open_time_naive",
        direction="backward",
        tolerance=pd.Timedelta("7D"),
    )

    # keep only what we need for selection (no lookahead)
    keep = ["engine","entry_time","exit_time","ret_pct","bull_on","close","ema50","ret1","bear_dip"]
    for c in keep:
        if c not in merged.columns:
            merged[c] = np.nan
    return merged[keep].copy()


def single_switch_backtest(trades: pd.DataFrame) -> dict:
    trades = trades.sort_values(["entry_time","engine"]).reset_index(drop=True)
    priority_rank = {e:i for i,e in enumerate(TIE_PRIORITY)}

    eq = 1.0
    peak = 1.0
    mdd = 0.0
    last_exit = pd.Timestamp.min
    min_gap = timedelta(minutes=MIN_REENTRY_MINUTES)

    used = []

    for t, grp in trades.groupby("entry_time", sort=True):
        if t < last_exit + min_gap:
            continue

        dd = eq/peak - 1.0

        best = None
        for _, row in grp.iterrows():
            engine = row["engine"]
            sc = quality_score(row)
            if sc <= SCORE_SKIP:
                continue

            base = base_dd_size(dd, peak, engine)
            if base <= 0:
                continue

            # score->quality multiplier (risk-on only when dd is shallow enough)
            q_mult = 1.0
            is_idx = str(engine).startswith("IDX")
            if is_idx and dd > DD_RISKON_ON and sc >= RISK_ON_SCORE_MIN:
                q_mult = RISK_ON_MULT
            elif is_idx and dd > DD_RISKON_ON and sc == (RISK_ON_SCORE_MIN - 1):
                q_mult = 1.13

            if dd <= DD_RISKON_CAP:
                q_mult = min(q_mult, 1.0)

            size = base * q_mult
            if size <= 0:
                continue

            pr = priority_rank.get(engine, 999)
            key = (sc, size, -pr)  # higher score, higher size, then priority
            if best is None or key > best["key"]:
                best = {"row": row, "sc": sc, "size": size, "key": key}

        if best is None:
            continue

        r = best["row"]
        scaled_ret = float(r["ret_pct"]) * float(best["size"])

        eq_before = eq
        eq = eq * (1.0 + scaled_ret)
        peak = max(peak, eq)
        mdd = min(mdd, eq/peak - 1.0)

        last_exit = r["exit_time"]

        used.append({
            "engine": r["engine"],
            "entry_time": r["entry_time"],
            "exit_time": r["exit_time"],
            "score": best["sc"],
            "size_mult": float(best["size"]),
            "ret_pct_raw": float(r["ret_pct"]),
            "ret_pct_scaled": float(scaled_ret),
            "dd_before": float(dd),
            "eq_before": float(eq_before),
            "eq_after": float(eq),
        })

    used_df = pd.DataFrame(used)
    trades_n = len(used_df)
    winrate = float((used_df["ret_pct_scaled"] > 0).mean() * 100.0) if trades_n else 0.0

    return {
        "Multiple": float(eq),
        "RealizedMDD": float(mdd),
        "Trades": int(trades_n),
        "WinRate_pct": float(winrate),
        "used": used_df
    }


def main():
    idx = load_module(IDX_ENGINE, "idx_engine")
    v42 = load_module(V42_ENGINE, "v42_engine")

    # IDX pipeline (30m derived from 15m + 12h macro + 1h micro)
    df15 = idx.load_15m(PATH_15M)
    df30 = idx.resample_30m_from_15m(df15)
    df1h = idx.load_1h(PATH_1H)
    macro12 = idx.build_macro_12h(df1h)
    feat1h = idx.build_micro_1h_features(df1h)
    df_idx = idx.merge_features(df30, macro12, feat1h)
    res_idx = idx.backtest_long_only(df_idx)

    # V42 pipeline
    df_v42 = v42.load_pipeline_6h30m15m("2021-01-01")
    res_v42 = v42.backtest_long(df_v42)

    idx_tr = enrich_trade_log(res_idx["trade_log"], "IDX500", df_idx)
    v42_tr = enrich_trade_log(res_v42["trade_log"], "V42", df_v42)

    trades = pd.concat([idx_tr, v42_tr], ignore_index=True)
    out = single_switch_backtest(trades)

    out["used"].to_csv(OUT_USED_TRADES, index=False)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump({key: out[key] for key in out.keys() if key!="used"}, f, ensure_ascii=False, indent=2)

    print("=== Single-switch + QualityGate(Q5) ===")
    print(f"Multiple: {out['Multiple']:.4f}x")
    print(f"Realized MDD: {out['RealizedMDD']*100:.2f}%")
    print(f"WinRate: {out['WinRate_pct']:.2f}%")
    print(f"Trades: {out['Trades']}")
    print(f"Saved: {OUT_USED_TRADES}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    args = parse_args()
    PATH_15M = _resolve(args.path_15m)
    PATH_1H  = _resolve(args.path_1h)
    IDX_ENGINE = _resolve(args.idx_engine)
    V42_ENGINE = _resolve(args.v42_engine)
    OUT_USED_TRADES = _resolve(args.out_used_trades)
    OUT_SUMMARY     = _resolve(args.out_summary)

    # Basic existence checks (fail fast with clear message)
    for _p in [PATH_15M, PATH_1H, IDX_ENGINE, V42_ENGINE]:
        if not os.path.exists(_p):
            raise FileNotFoundError(f'Missing file: {_p}')

    main()



# =========================
# Post-run MTM MDD (15m OHLC worst excursion, long-only approximation)
# - Uses candle LOW between entry_time..exit_time as worst adverse move
# - Computes portfolio-level MTM drawdown vs running peak (chronological)
# NOTE: If your engines can go short, extend this with side information.
# =========================
def compute_mtm_mdd_15m(used_trades: pd.DataFrame, df15: pd.DataFrame) -> float:
    df15 = df15.copy()
    if "datetime" in df15.columns:
        df15["datetime"] = pd.to_datetime(df15["datetime"])
        df15 = df15.sort_values("datetime").set_index("datetime")
    else:
        raise ValueError("15m data must contain 'datetime' column.")
    used_trades = used_trades.copy()
    used_trades["entry_time"] = pd.to_datetime(used_trades["entry_time"])
    used_trades["exit_time"]  = pd.to_datetime(used_trades["exit_time"])
    used_trades = used_trades.sort_values("entry_time")

    def close_at(ts):
        ts = pd.to_datetime(ts)
        if ts in df15.index:
            return float(df15.loc[ts, "close"])
        # use last available bar <= ts
        idx = df15.index[df15.index.get_indexer([ts], method="pad")][0]
        return float(df15.loc[idx, "close"])

    peak = 1.0
    mtm_mdd = 0.0
    for _, r in used_trades.iterrows():
        et = r["entry_time"]; xt = r["exit_time"]
        eq_before = float(r["eq_before"])
        size = float(r["size_mult"])
        peak = max(peak, eq_before)

        entry_px = close_at(et)
        seg = df15.loc[et:xt]
        if len(seg) == 0:
            continue

        min_low = float(seg["low"].min())
        worst_ret = (min_low / entry_px) - 1.0  # long-only adverse move
        worst_eq = eq_before * (1.0 + size * worst_ret)
        dd = (worst_eq / peak) - 1.0
        mtm_mdd = min(mtm_mdd, dd)

        eq_after = float(r["eq_after"])
        peak = max(peak, eq_after)

    return float(mtm_mdd)


# After saving OUT_USED_TRADES and OUT_SUMMARY, compute MTM and update summary json.
try:
    used_df = pd.read_csv(OUT_USED_TRADES)
    df15m_local = pd.read_csv(PATH_15M)
    mtm_mdd_15m = compute_mtm_mdd_15m(used_df, df15m_local)
    # patch summary
    with open(OUT_SUMMARY, "r", encoding="utf-8") as f:
        summ = json.load(f)
    summ["MTM_MDD_15m_longonly"] = mtm_mdd_15m
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(f"MTM MDD (15m low, long-only approx): {mtm_mdd_15m*100:.2f}%")
except Exception as e:
    print("MTM MDD post-calc skipped due to:", repr(e))