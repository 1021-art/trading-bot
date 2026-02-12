#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CALL 싱글스위치 + PUT 라우터 — SWITCH0(STRICT 비겹침) 포트폴리오 + 정밀 MTM

입력(현재 업로드 파일 기준)
- 15m 데이터: Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv
- CALL(롱) 단: _single_used.csv  (single_switch_D_... 실행 결과)
- PUT(숏)  단: _router_trades.csv (short_router_crash_plus_lite_... 실행 결과)

SWITCH0 규칙
- entry_time 기준 정렬 후, 현재 포지션이 열려있는 동안(entry_time < cur_end) 진입은 DROP
- 동시간 entry_time이면 CALL 우선

정밀 MTM 정의 (사용자 정의)
- LONG: entry~exit 구간 15m low 최저값으로 adverse move 계산
- SHORT: entry~exit 구간 15m high 최고값으로 adverse move 계산
- 포트폴리오 MTM: peak equity 대비 intratrade worst equity의 최저 낙폭

주의
- 이 스크립트는 "백테스트 사후" 분석으로 MTM을 계산합니다(미래정보로 의사결정 X).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent


def to_utc_naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def compute_realized_mdd(eq_points) -> float:
    peak = -np.inf
    mdd = 0.0
    for e in eq_points:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1.0)
    return mdd


def build_switch0_tape(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    raw = raw.copy()
    raw["tie_rank"] = raw["engine"].map({"CALL": 0, "PUT": 1}).fillna(9).astype(int)
    raw = raw.sort_values(["entry_time", "tie_rank", "exit_time"]).reset_index(drop=True)

    kept = []
    dropped = 0
    cur_end = None

    for _, r in raw.iterrows():
        if cur_end is None:
            kept.append(r)
            cur_end = r["exit_time"]
            continue

        if r["entry_time"] < cur_end:
            dropped += 1
            continue

        kept.append(r)
        cur_end = r["exit_time"]

    tape = pd.DataFrame(kept).drop(columns=["tie_rank"], errors="ignore")
    return tape.sort_values(["entry_time", "exit_time"]).reset_index(drop=True), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_15m", default=str(BASE / "Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv"))
    ap.add_argument("--call_used", default=str(BASE / "_single_used.csv"))
    ap.add_argument("--put_trades", default=str(BASE / "_router_trades.csv"))
    ap.add_argument("--out_tape", default=str(BASE / "CALLSW_PUT_master_tape_SWITCH0.csv"))
    ap.add_argument("--out_summary", default=str(BASE / "CALLSW_PUT_portfolio_summary_SWITCH0_MTM.txt"))
    args = ap.parse_args()

    # 15m
    df15 = pd.read_csv(args.path_15m)
    if "datetime" not in df15.columns:
        raise ValueError("15m CSV must have 'datetime' column")
    df15["datetime"] = pd.to_datetime(df15["datetime"])
    df15 = df15.sort_values("datetime").set_index("datetime")

    def close_at(ts: pd.Timestamp) -> float:
        ts = to_utc_naive(ts)
        if ts not in df15.index:
            # nearest previous bar
            ix = df15.index.searchsorted(ts, side="right") - 1
            if ix < 0:
                raise KeyError(f"No bar <= {ts}")
            ts2 = df15.index[ix]
            return float(df15.loc[ts2, "close"])
        return float(df15.loc[ts, "close"])

    # CALL used trades
    call = pd.read_csv(args.call_used)
    call["entry_time"] = pd.to_datetime(call["entry_time"]).apply(to_utc_naive)
    call["exit_time"] = pd.to_datetime(call["exit_time"]).apply(to_utc_naive)
    # ret_pct_scaled is decimal return already size-adjusted
    if "ret_pct_scaled" not in call.columns or "size_mult" not in call.columns:
        raise ValueError("CALL used trades must include ret_pct_scaled and size_mult")
    call_df = call[["entry_time", "exit_time", "ret_pct_scaled", "size_mult", "engine"]].copy()
    call_df = call_df.rename(columns={"ret_pct_scaled": "ret_pct"})
    call_df["engine"] = "CALL"

    # PUT trades
    put = pd.read_csv(args.put_trades)
    put["entry_time"] = pd.to_datetime(put["entry_time"]).apply(to_utc_naive)
    put["exit_time"] = pd.to_datetime(put["exit_time"]).apply(to_utc_naive)
    if "ret_pct" not in put.columns:
        raise ValueError("PUT trades must include ret_pct")
    put_df = put[["entry_time", "exit_time", "ret_pct", "engine"]].copy()
    put_df["engine"] = "PUT"
    put_df["size_mult"] = 1.0

    raw = pd.concat([call_df, put_df], ignore_index=True)
    tape, dropped = build_switch0_tape(raw)

    # realized curve
    eq = 1.0
    peak = 1.0
    eq_points = [eq]
    mtm_mdd = 0.0
    mtm_worst = None

    rows = []

    for i, r in tape.iterrows():
        et, xt = r["entry_time"], r["exit_time"]
        ret = float(r["ret_pct"])
        size = float(r.get("size_mult", 1.0))
        eng = r["engine"]

        eq_before = eq
        peak = max(peak, eq_before)

        entry_px = close_at(et)
        seg = df15.loc[et:xt]
        if len(seg) == 0:
            worst_eq = eq_before
        else:
            if eng == "CALL":
                # long adverse: min low
                min_low = float(seg["low"].min())
                worst_ret = (min_low / entry_px) - 1.0
                worst_eq = eq_before * (1.0 + size * worst_ret)
            else:
                # short adverse: max high
                max_high = float(seg["high"].max())
                worst_ret = (entry_px / max_high) - 1.0  # <=0
                worst_eq = eq_before * (1.0 + worst_ret)  # size=1 for put

        dd = (worst_eq / peak) - 1.0
        if dd < mtm_mdd:
            mtm_mdd = dd
            mtm_worst = (i, eng, et, xt, dd)

        # apply realized return
        if eng == "CALL":
            eq = eq_before * (1.0 + size * ret)
        else:
            eq = eq_before * (1.0 + ret)

        eq_points.append(eq)

        rows.append({
            "engine": eng,
            "entry_time": et,
            "exit_time": xt,
            "ret_pct": ret,
            "size_mult": size,
            "eq_before": eq_before,
            "eq_after": eq,
            "peak_before": peak,
            "worst_eq_intrade": worst_eq,
            "mtm_dd_intrade": dd,
        })

    tape_out = pd.DataFrame(rows)
    tape_out.to_csv(args.out_tape, index=False)

    multiple = eq
    realized_mdd = compute_realized_mdd(eq_points)
    winrate = (tape_out["ret_pct"] > 0).mean() * 100.0 if len(tape_out) else 0.0

    lines = []
    lines.append("CALL(싱글스위치) + PUT(라우터) Portfolio — SWITCH0(STRICT non-overlap)")
    lines.append("=" * 78)
    lines.append(f"Trades: {len(tape_out)}  | Dropped(overlap): {dropped}")
    lines.append(f"Win-rate: {winrate:.2f}%")
    lines.append(f"Multiple: {multiple:.4f}x")
    lines.append(f"Realized MDD: {realized_mdd*100.0:.4f}%")
    lines.append(f"MTM MDD(정밀): {mtm_mdd*100.0:.4f}%")

    if mtm_worst is not None:
        i, eng, et, xt, dd = mtm_worst
        lines.append("")
        lines.append("MTM Worst trade:")
        lines.append(f" - idx: {i}")
        lines.append(f" - engine: {eng}")
        lines.append(f" - entry: {et}")
        lines.append(f" - exit : {xt}")
        lines.append(f" - dd   : {dd*100.0:.4f}%")

    Path(args.out_summary).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
