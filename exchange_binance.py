"""Binance USDT-M Futures HTTP client with DRY/TESTNET/LIVE modes."""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests


class BinanceUSDMExchange:
    def __init__(self, api_key: str, api_secret: str, mode: str = "DRY") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode.upper()
        if self.mode == "TESTNET":
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"

    @property
    def is_dry(self) -> bool:
        return self.mode == "DRY"

    def _public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def _signed_request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if self.is_dry:
            raise RuntimeError("Signed request is unavailable in DRY mode")
        params = params.copy() if params else {}
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        query = f"{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{path}?{query}"
        response = requests.request(method, url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
        return self._public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def get_mark_price(self, symbol: str) -> float:
        data = self._public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(data["markPrice"])

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        if self.is_dry:
            return {"mode": "DRY", "symbol": symbol, "leverage": leverage}
        return self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.3f}",
            "reduceOnly": "true" if reduce_only else "false",
        }
        if self.is_dry:
            return {"mode": "DRY", "action": "place_market_order", **payload}
        return self._signed_request("POST", "/fapi/v1/order", payload)
