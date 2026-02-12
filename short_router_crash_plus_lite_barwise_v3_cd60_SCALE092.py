
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar-wise SHORT router (event-driven, position-exclusive) — SCALE092 preset
-------------------------------------------------------
- Runs two short engines (CrashShort + LiteShort) on the same market data.
- Builds each engine's own feature frame (using functions inside each engine file).
- Executes BOTH backtests to obtain their trade logs (entry_time/exit_time/ret_pct).
- Then re-simulates a SINGLE account that can hold at most ONE short position at a time.

Routing policy:
1) If no position is open and both engines signal an entry at the same timestamp:
   -> Crash engine wins (priority).
2) If a position is open, any candidate entries with entry_time < current_exit_time are ignored.
3) Optional global cooldown after any exit can be applied (minutes).

This avoids overlap in a realistic way without re-implementing each engine's intrabar logic.
No lookahead: each engine is responsible for its own no-lookahead feature merge.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class Trade:
    engine: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    ret_pct: float
    reason: Any

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _prepare_df(engine_mod, path_15m: str, path_1h: str) -> pd.DataFrame:
    # Engines in this project expose these functions.
    df15 = engine_mod.load_15m(path_15m)
    df30 = engine_mod.resample_30m_from_15m(df15)
    df1h = engine_mod.load_1h(path_1h)
    macro12 = engine_mod.build_macro_12h(df1h)
    feat1h = engine_mod.build_micro_1h_features(df1h)
    df = engine_mod.merge_features(df30, macro12, feat1h)
    return df

def _run_engine(engine_mod, df: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # backtest_short_only returns summary dict; get_tradelog returns list of trades
    summary = engine_mod.backtest_short_only(df)
    trades = engine_mod.get_tradelog()
    if not isinstance(trades, list):
        raise TypeError("Engine get_tradelog() must return a list")
    return summary, trades

def _as_trade_list(trades_raw: List[Dict[str, Any]], engine_name: str) -> List[Trade]:
    out: List[Trade] = []
    for t in trades_raw:
        try:
            out.append(Trade(
                engine=engine_name,
                entry_time=pd.Timestamp(t["entry_time"]),
                exit_time=pd.Timestamp(t["exit_time"]),
                ret_pct=float(t["ret_pct"]),
                reason=t.get("reason", None),
            ))
        except Exception as e:
            raise ValueError(f"Bad trade format in {engine_name}: {t}") from e
    out.sort(key=lambda x: (x.entry_time, x.exit_time))
    return out

def _simulate_one_account(trades_a: List[Trade], trades_b: List[Trade],
                          priority_engine: str,
                          global_cooldown_minutes: int = 0,
                          start_equity: float = 10_000_000.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Position-exclusive simulation using trade logs.
    Equity is compounded at each trade close: eq *= (1 + ret_pct).
    Realized MDD computed on equity after each trade close.
    """
    i = j = 0
    eq = start_equity
    peak = start_equity
    mdd = 0.0
    wins = 0
    ntr = 0

    current_free_time = None  # when we can enter next trade (exit time + cooldown)

    used: List[Dict[str, Any]] = []

    def pick_next(i: int, j: int) -> Optional[Trade]:
        cand: List[Trade] = []
        if i < len(trades_a): cand.append(trades_a[i])
        if j < len(trades_b): cand.append(trades_b[j])
        if not cand:
            return None
        # earliest entry_time first
        cand.sort(key=lambda x: (x.entry_time, 0 if x.engine == priority_engine else 1))
        # If tie on entry_time, priority_engine comes first via secondary key
        return cand[0]

    while True:
        nxt = pick_next(i, j)
        if nxt is None:
            break

        # advance pointer for chosen trade
        if i < len(trades_a) and trades_a[i] is nxt:
            i += 1
        elif j < len(trades_b) and trades_b[j] is nxt:
            j += 1
        else:
            # Shouldn't happen
            raise RuntimeError("Pointer mismatch")

        # enforce position exclusivity / cooldown
        if current_free_time is not None and nxt.entry_time < current_free_time:
            continue

        # take trade
        eq_before = eq
        eq = eq * (1.0 + nxt.ret_pct)
        ntr += 1
        if nxt.ret_pct > 0:
            wins += 1

        peak = max(peak, eq)
        dd = eq / peak - 1.0
        mdd = min(mdd, dd)

        # update next free time
        free_time = nxt.exit_time
        if global_cooldown_minutes > 0:
            free_time = free_time + pd.Timedelta(minutes=global_cooldown_minutes)
        current_free_time = free_time

        used.append({
            "engine": nxt.engine,
            "entry_time": nxt.entry_time.isoformat(),
            "exit_time": nxt.exit_time.isoformat(),
            "ret_pct": nxt.ret_pct,
            "reason": nxt.reason,
            "eq_before": eq_before,
            "eq_after": eq,
            "dd_after": dd,
        })

    used_df = pd.DataFrame(used)
    multiple = eq / start_equity if start_equity > 0 else float("nan")
    winrate = wins / ntr if ntr > 0 else float("nan")

    summary = {
        "start_equity": start_equity,
        "end_equity": eq,
        "multiple": multiple,
        "realized_mdd": mdd,  # negative
        "trades": ntr,
        "winrate": winrate,
        "wins": wins,
        "global_cooldown_minutes": global_cooldown_minutes,
        "priority_engine": priority_engine,
        "note": "realized_mdd is computed on equity at trade closes (trade-log simulation).",
    }
    return used_df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_15m", required=True)
    ap.add_argument("--path_1h", required=True)
    ap.add_argument("--crash_engine", default="/mnt/data/IDX500_SHORT_V52V2_RISK015_SCALE_ATRP030_092.py")
    ap.add_argument("--lite_engine", default="/mnt/data/IDX500_SHORT_LITE_Q_Bear0Ret017_RON14_M130.py")
    ap.add_argument("--priority", default="CRASH", choices=["CRASH", "LITE"])
    ap.add_argument("--global_cooldown_minutes", type=int, default=60)
    ap.add_argument("--out_trades", default="combined_used_trades_barwise.csv")
    ap.add_argument("--out_summary", default="combined_summary_barwise.json")
    args = ap.parse_args()

    crash_mod = _load_module(args.crash_engine, "crash_engine_mod")
    lite_mod = _load_module(args.lite_engine, "lite_engine_mod")

    # Build each engine's feature frame
    df_crash = _prepare_df(crash_mod, args.path_15m, args.path_1h)
    df_lite = _prepare_df(lite_mod, args.path_15m, args.path_1h)

    # Run each engine
    _, crash_trades_raw = _run_engine(crash_mod, df_crash)
    _, lite_trades_raw = _run_engine(lite_mod, df_lite)

    crash_trades = _as_trade_list(crash_trades_raw, "CRASH")
    lite_trades = _as_trade_list(lite_trades_raw, "LITE")

    used_df, summary = _simulate_one_account(
        crash_trades, lite_trades,
        priority_engine=args.priority,
        global_cooldown_minutes=args.global_cooldown_minutes
    )

    used_df.to_csv(args.out_trades, index=False, encoding="utf-8")
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
