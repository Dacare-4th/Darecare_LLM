# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils/currency.py
# 역할 : 실시간 환율 조회 및 보험금 KRW 환산 계산
#
# API  : ExchangeRate-API (https://www.exchangerate-api.com)
#        무료 플랜 기준 1,500 req/월
#        EXCHANGE_RATE_API_KEY 환경변수 필요
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os
import time
from functools import lru_cache

import requests

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

# ExchangeRate-API 엔드포인트 (기준 통화 → 전체 환율 조회)
_API_BASE = "https://api.frankfurter.dev/v2/rates"

# 기준 통화 (KRW 으로 환산)
_BASE_CURRENCY = "KRW"

# 캐시 TTL (초) — 실시간이지만 동일 세션 내 중복 호출 방지
_CACHE_TTL = 600   # 10분

# 캐시 저장소 {currency: (rate, timestamp)}
_rate_cache: dict[str, tuple[float, float]] = {}

# 지원 통화 코드
SUPPORTED_CURRENCIES = {
    "USD", "EUR", "JPY", "GBP", "CNY", "CHF",
    "CAD", "AUD", "SGD", "HKD", "THB",
}


# ──────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────

def get_exchange_rate(currency: str, rate_date:str="") -> float:
    """
    지정 통화 → KRW 환율을 조회한다. (1 [currency] = ? KRW)

    캐시가 유효하면 캐시 값을 반환한다.
    API 키 없이 사용 가능하며, rate_date가 있으면 해당 날짜 환율을 조회한다.

    Args:
        currency: 통화 코드 (예: "USD", "EUR", "JPY")
        rate_date: 환율 기준일 (예: "2025-03-15", 없으면 최신 환율)

    Returns:
        환율 (float)
        오류 시 fallback 환율 반환
    """
    currency = currency.upper().strip() if currency else ""
    if not currency:
        print("[currency] 환율 조회 실패: 통화 코드가 없습니다.")
        return 0.0

    cache_key = f"{currency}:{rate_date or 'latest'}"
    cached = _rate_cache.get(cache_key)
    if cached:
        rate, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return rate

    try:
        params = {
            "base": currency,
            "quotes": "KRW",
        }
        if rate_date:
            params["date"] = rate_date

        resp = requests.get(_API_BASE, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        krw_row = next((item for item in data if item.get("quote") == "KRW"), None)
        if not krw_row:
            raise ValueError("KRW 환율 데이터가 응답에 없습니다.")
        rate = float(krw_row["rate"])
        _rate_cache[cache_key] = (rate, time.time())    
        return rate
        
    except Exception as e:
        print(f"[currency] 환율 API 오류:{type(e).__name__} - {e}")
        return _fallback_rate(currency)


def convert_to_krw(amount: float, currency: str, rate_date:str ="" ) -> dict:
    """
    외화 금액을 KRW 로 환산한다.

    Args:
        amount  : 변환할 금액 (외화 기준)
        currency: 원본 통화 코드 (예: "USD")

    Returns:
        {
            "original_amount": float,
            "currency"       : str,
            "exchange_rate"  : float,   # 1 [currency] = ? KRW
            "amount_krw"     : float,   # 환산 금액 (KRW)
        }
    """
    rate = get_exchange_rate(currency,  rate_date)
    return {
        "original_amount": amount,
        "currency"       : currency.upper(),
        "exchange_rate"  : rate,
        "rate_date": rate_date or "latest",
        "amount_krw"     : round(amount * rate, 0),
    }


def calculate_copay(
    total_amount: float,
    currency: str,
    deductible: float = 0.0,
    copay_rate: float = 0.2,
    rate_date:str="",
) -> dict:
    """
    본인부담금을 계산하고 KRW 로 환산한다.

    계산식:
        본인부담금 = max(total_amount - deductible, 0) × copay_rate
        보험 청구 가능액 = total_amount - deductible - 본인부담금

    Args:
        total_amount: 총 의료비 (외화)
        currency    : 통화 코드 (예: "USD")
        deductible  : 공제액 — 보험이 적용되기 전 본인이 먼저 내는 금액 (외화)
        copay_rate  : 공동부담률 — 공제 후 본인이 부담하는 비율 (0.0 ~ 1.0)
                      예: 0.2 = 20% 본인부담

    Returns:
        {
            "total_amount"    : float,  # 총 의료비 (외화)
            "currency"        : str,
            "deductible"      : float,  # 공제액 (외화)
            "copay_rate"      : float,
            "copay_amount"    : float,  # 본인부담금 (외화)
            "claimable_amount": float,  # 보험 청구 가능액 (외화)
            "exchange_rate"   : float,  # 환율
            "copay_krw"       : float,  # 본인부담금 (KRW)
            "claimable_krw"   : float,  # 청구 가능액 (KRW)
        }
    """
    after_deductible  = max(total_amount - deductible, 0.0)
    copay_amount      = round(after_deductible * copay_rate, 2)
    claimable_amount  = round(after_deductible - copay_amount, 2)

    rate              = get_exchange_rate(currency, rate_date)

    return {
        "total_amount"    : total_amount,
        "currency"        : currency.upper(),
        "deductible"      : deductible,
        "copay_rate"      : copay_rate,
        "copay_amount"    : copay_amount,
        "claimable_amount": claimable_amount,
        "rate_date": rate_date or "latest",
        "exchange_rate"   : rate,
        "copay_krw"       : round(copay_amount    * rate, 0),
        "claimable_krw"   : round(claimable_amount * rate, 0),
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _fallback_rate(currency: str) -> float:
    """
    API 를 사용할 수 없을 때 참조하는 근사 환율.
    실제 서비스에서는 API 키를 반드시 설정할 것.
    """
    fallback = {
        "USD": 1350.0,
        "EUR": 1480.0,
        "JPY": 9.0,
        "GBP": 1720.0,
        "CNY": 186.0,
        "CHF": 1530.0,
        "CAD": 990.0,
        "AUD": 880.0,
        "SGD": 1010.0,
        "HKD": 173.0,
        "THB": 38.0,
    }
    return fallback.get(currency.upper(), 0.0)
