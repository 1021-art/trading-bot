"""Configuration loader for Binance live controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    mode: str
    symbol: str
    interval_signal: str
    interval_risk: str
    leverage: int
    qty_usdt: float
    exit1_notional_cap_krw: float
    krw_per_usdt: float
    api_key: str
    api_secret: str
    poll_seconds: int
    state_path: str
    ema_fast: int
    ema_slow: int

    @property
    def is_dry(self) -> bool:
        return self.mode == "DRY"

    @property
    def use_testnet(self) -> bool:
        return self.mode == "TESTNET"

    @property
    def exit1_notional_cap_usdt(self) -> float:
        if self.krw_per_usdt <= 0:
            return self.exit1_notional_cap_krw
        return self.exit1_notional_cap_krw / self.krw_per_usdt


def _required_env(name: str, *, allow_empty_in_dry: bool = False) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    mode = os.getenv("MODE", "DRY").upper().strip()
    if allow_empty_in_dry and mode == "DRY":
        return ""
    raise ValueError(f"Missing required environment variable: {name}")


def load_settings() -> Settings:
    mode = os.getenv("MODE", "DRY").upper().strip()
    if mode not in {"DRY", "TESTNET", "LIVE"}:
        raise ValueError("MODE must be one of DRY/TESTNET/LIVE")

    return Settings(
        mode=mode,
        symbol=os.getenv("SYMBOL", "BTCUSDT").upper().strip(),
        interval_signal=os.getenv("INTERVAL_SIGNAL", "15m").strip(),
        interval_risk=os.getenv("INTERVAL_RISK", "1h").strip(),
        leverage=int(os.getenv("LEVERAGE", "2")),
        qty_usdt=float(os.getenv("QTY_USDT", "5000")),
        exit1_notional_cap_krw=float(os.getenv("EXIT1_NOTIONAL_CAP_KRW", "800000000")),
        krw_per_usdt=float(os.getenv("KRW_PER_USDT", "1300")),
        api_key=_required_env("BINANCE_API_KEY", allow_empty_in_dry=True),
        api_secret=_required_env("BINANCE_API_SECRET", allow_empty_in_dry=True),
        poll_seconds=int(os.getenv("POLL_SECONDS", "10")),
        state_path=os.getenv("STATE_PATH", "state_live_binance.json"),
        ema_fast=int(os.getenv("EMA_FAST", "20")),
        ema_slow=int(os.getenv("EMA_SLOW", "50")),
    )
