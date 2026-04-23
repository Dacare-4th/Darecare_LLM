# frankfurter.app 호출 — KRW/USD/EUR 등 환율 변환
import requests


def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """최신 환율 반환 (API 키 불필요)"""
    url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"
    res = requests.get(url, timeout=5)
    data = res.json()
    return data["rates"][to_currency]


def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """금액 환율 변환"""
    rate = get_exchange_rate(from_currency, to_currency)
    return round(amount * rate, 2)
