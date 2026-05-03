# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils/language.py
# 역할 : 사용자 질의의 언어를 감지한다.
#
# 전략:
#   0. 고유명사/전문용어 제거
#   1. Unicode 스크립트 비율로 비라틴계 언어 감지 (ko/ja/zh)
#   2. langdetect (라틴계 언어 구분)
#   3. LLM fallback (혼합/애매한 경우)
# 지원 언어 : 한국어·영어·일본어·중국어·프랑스어·독일어·스페인어
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os
import re
from langdetect import detect_langs
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

SUPPORTED_LANGS = {"ko", "en", "ja", "zh-cn", "zh-tw", "fr", "de", "es"}

LANG_NORMALIZE = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}

CONFIDENCE_THRESHOLD = 0.85

# 비라틴계 스크립트 감지 임계값 (전체 알파벳 문자 중 비율)
SCRIPT_THRESHOLD = 0.15

# ── 고유명사/보험 전문용어 목록 ────────────────────────────────
# 언어 감지 전 제거할 단어들 (대소문자 무관)
_PROPER_NOUNS = {
    # 보험사
    "cigna", "tricare", "uhcg", "uhc", "unitedhealth",
    "msh", "msh china", "nhis",
    # 플랜명
    "gold", "silver", "platinum", "pearl", "sapphire", "diamond",
    "quartz", "hospi", "health", "health+", "health+child",
    "core", "prime", "select", "standard", "basic",
    # 의료/보험 영어 전문용어
    "referral", "authorization", "pre-authorization", "preauthorization",
    "copay", "deductible", "premium", "reimbursement", "claim",
    "coverage", "benefit", "plan", "policy", "enrollment",
    "inpatient", "outpatient", "dental", "vision", "maternity",
    "oconus", "usfk", "tricare4u",
    # 지역/기관
    "korea", "seoul", "usa", "oconus",
}

_PROPER_NOUN_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_PROPER_NOUNS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ── LLM fallback 프롬프트 ──────────────────────────────────────
_LANG_SYSTEM_PROMPT = """You are a language detector for a multilingual insurance chatbot.
The text may contain English insurance/medical terms regardless of the user's language.
Identify the PRIMARY language the USER intends to communicate in, ignoring technical terms and proper nouns.

Respond with ONLY one of these codes:
ko (Korean), en (English), ja (Japanese), zh (Chinese),
fr (French), de (German), es (Spanish).
If uncertain, respond with "en".
Respond with the code only, no explanation."""


# ──────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    텍스트의 주 언어를 감지한다.

    0단계: 고유명사/전문용어 제거
    1단계: Unicode 스크립트 비율 → ko/ja/zh 판별
    2단계: langdetect → 라틴계 언어 판별
    3단계: LLM fallback → 혼합/애매한 경우
    """
    if not text or not text.strip():
        return "en"

    # ── 0단계: 고유명사 제거 후 분석용 텍스트 생성 ────────────
    clean = _PROPER_NOUN_RE.sub(" ", text).strip()
    if not clean:
        clean = text  # 전부 제거된 경우 원문 사용

    # ── 1단계: 스크립트 비율 감지 ─────────────────────────────
    script_lang = _detect_by_script(clean)
    if script_lang:
        return script_lang

    # ── 2단계: langdetect ──────────────────────────────────────
    try:
        results = detect_langs(clean)
        top     = results[0]
        lang    = LANG_NORMALIZE.get(top.lang, top.lang)
        prob    = top.prob

        if lang in SUPPORTED_LANGS and prob >= CONFIDENCE_THRESHOLD:
            return lang

    except Exception:
        pass

    # ── 3단계: LLM fallback ────────────────────────────────────
    return _llm_detect(text)  # 원문 그대로 전달


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _detect_by_script(text: str) -> str | None:
    """
    Unicode 스크립트 블록 비율로 비라틴계 언어를 감지한다.

    한글/히라가나+카타카나/CJK 각각의 비율이 SCRIPT_THRESHOLD 이상이면
    해당 언어를 반환한다. 여러 개 해당되면 가장 높은 비율의 언어를 반환.
    """
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return None

    total = len(alpha)

    scores: dict[str, float] = {
        "ko": sum(1 for ch in alpha if "가" <= ch <= "힣") / total,
        "ja": sum(1 for ch in alpha if "぀" <= ch <= "ヿ") / total,
        "zh": sum(1 for ch in alpha if "一" <= ch <= "鿿") / total,
    }

    # 임계값 이상인 언어 중 가장 높은 것 반환
    candidates = {lang: ratio for lang, ratio in scores.items() if ratio >= SCRIPT_THRESHOLD}
    if candidates:
        return max(candidates, key=lambda k: candidates[k])

    return None


def _llm_detect(text: str) -> str:
    """LLM으로 언어를 감지한다. 실패 시 "en" 반환."""
    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": _LANG_SYSTEM_PROMPT},
                {"role": "user",   "content": text[:300]},
            ],
            max_tokens  = 5,
            temperature = 0,
        )
        lang = response.choices[0].message.content.strip().lower()

        if lang in {"ko", "en", "ja", "zh", "fr", "de", "es"}:
            return lang

    except Exception:
        pass

    return "en"
