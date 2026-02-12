"""Live/Sim controller for Binance USDT-M Futures.

Rules implemented:
- MODE=DRY/TESTNET/LIVE
- Signal computed only at closed 15m candle (no lookahead)
- If stop and TP are both touched in same 1h candle, STOP is prioritized
- No re-entry within the same 1h candle after an exit
- Exit1 notional cap is enforced at 800,000,000 KRW by default
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import load_settings
from exchange_binance import BinanceUSDMExchange
from state_store import StateStore


def ts_to_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def compute_signal(closes: List[float], fast: int, slow: int) -> str:
    if len(closes) < max(fast, slow) + 2:
        return "FLAT"
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    prev_fast, curr_fast = fast_ema[-2], fast_ema[-1]
    prev_slow, curr_slow = slow_ema[-2], slow_ema[-1]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return "LONG"
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return "SHORT"
    return "FLAT"


def safe_qty(notional_usdt: float, mark_price: float) -> float:
    if mark_price <= 0:
        return 0.0
    qty = notional_usdt / mark_price
    return math.floor(qty * 1000) / 1000


def _current_1h_open_ms(now_ms: int) -> int:
    hour_ms = 60 * 60 * 1000
    return now_ms - (now_ms % hour_ms)


def check_hourly_exit_priority(position: Dict, hourly_kline: List) -> Optional[str]:
    """Return STOP/TP/None with enforced STOP->TP order in same candle."""
    high = float(hourly_kline[2])
    low = float(hourly_kline[3])
    side = position["side"]
    stop = position["stop_price"]
    take = position["take_profit_price"]

    if side == "LONG":
        stop_hit = low <= stop
        tp_hit = high >= take
    else:
        stop_hit = high >= stop
        tp_hit = low <= take

    if stop_hit:
        return "STOP"
    if tp_hit:
        return "TP"
    return None


def main() -> None:
    settings = load_settings()
    store = StateStore(settings.state_path)
    state = store.load()

    ex = BinanceUSDMExchange(settings.api_key, settings.api_secret, settings.mode)
    ex.set_leverage(settings.symbol, settings.leverage)

    print(f"[BOOT] MODE={settings.mode} SYMBOL={settings.symbol} POLL={settings.poll_seconds}s")
    print(f"[BOOT] Exit1 cap={settings.exit1_notional_cap_krw:,.0f} KRW ({settings.exit1_notional_cap_usdt:,.2f} USDT)")

    while True:
        try:
            k15 = ex.get_klines(settings.symbol, settings.interval_signal, limit=300)
            closed_15m = k15[-2]
            open_ms_15 = int(closed_15m[0])

            if state.get("last_15m_open_time") == open_ms_15:
                time.sleep(settings.poll_seconds)
                continue

            closes = [float(x[4]) for x in k15[:-1]]
            signal = compute_signal(closes, settings.ema_fast, settings.ema_slow)
            price = float(closes[-1])

            print(f"[15m CLOSE] {ts_to_text(open_ms_15)} signal={signal} price={price}")
            state["last_15m_open_time"] = open_ms_15

            # Process 1h risk candle exactly once per closed hour candle.
            k1h = ex.get_klines(settings.symbol, settings.interval_risk, limit=3)
            closed_1h = k1h[-2]
            hour_open_ms = int(closed_1h[0])
            if state.get("position") and state.get("last_processed_1h_open_time") != hour_open_ms:
                reason = check_hourly_exit_priority(state["position"], closed_1h)
                state["last_processed_1h_open_time"] = hour_open_ms
                if reason:
                    pos = state["position"]
                    close_side = "SELL" if pos["side"] == "LONG" else "BUY"
                    ex.place_market_order(settings.symbol, close_side, pos["qty"], reduce_only=True)
                    print(f"[EXIT] {reason} side={pos['side']} qty={pos['qty']}")
                    state["position"] = None
                    state["last_exit_hour_open_time"] = hour_open_ms

            # Re-entry restriction: do not re-enter in the same hour of the last exit.
            now_hour = _current_1h_open_ms(int(time.time() * 1000))
            same_hour_reentry_blocked = state.get("last_exit_hour_open_time") == now_hour

            if not state.get("position") and not same_hour_reentry_blocked and signal in {"LONG", "SHORT"}:
                mark_price = ex.get_mark_price(settings.symbol)
                allowed_notional = min(settings.qty_usdt, settings.exit1_notional_cap_usdt)
                qty = safe_qty(allowed_notional, mark_price)
                if qty > 0:
                    side = "BUY" if signal == "LONG" else "SELL"
                    ex.place_market_order(settings.symbol, side, qty, reduce_only=False)

                    stop_pct = 0.007
                    tp_pct = 0.010
                    if signal == "LONG":
                        stop_price = mark_price * (1 - stop_pct)
                        tp_price = mark_price * (1 + tp_pct)
                    else:
                        stop_price = mark_price * (1 + stop_pct)
                        tp_price = mark_price * (1 - tp_pct)

                    state["position"] = {
                        "side": signal,
                        "entry_price": mark_price,
                        "qty": qty,
                        "stop_price": stop_price,
                        "take_profit_price": tp_price,
                        "opened_15m": open_ms_15,
                    }
                    print(f"[ENTRY] side={signal} qty={qty} mark={mark_price}")
                else:
                    print("[SKIP] Computed qty is zero.")
            elif same_hour_reentry_blocked:
                print("[BLOCK] Re-entry blocked within same 1h candle.")

            store.save(state)
            time.sleep(settings.poll_seconds)

        except KeyboardInterrupt:
            print("[STOP] Interrupted by user.")
            store.save(state)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] loop error: {exc}")
            store.save(state)
            time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    main()
