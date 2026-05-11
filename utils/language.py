# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils/language.py
# 역할 : 사용자 질의의 언어를 감지한다.
#
# 전략:
#   0. 고유명사/전문용어 제거
#   1. Unicode 스크립트 비율로 비라틴계 언어 감지 (ko/ja/zh)
#   2. 라틴계 전용 문자로 de/es/fr 확정 감지
#   3. langdetect (라틴계 언어 구분, 신뢰도 기반)
#   4. LLM fallback (혼합/애매한 경우)
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

CONFIDENCE_THRESHOLD      = 0.85  # langdetect 높은 신뢰도
CONFIDENCE_THRESHOLD_LOW  = 0.55  # 악센트 문자로 보완 시 허용 하한

# 비라틴계 스크립트 감지 임계값 (전체 알파벳 문자 중 비율)
SCRIPT_THRESHOLD = 0.10

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
    # 보험 일반 영어 명사 (언어 감지 왜곡 방지)
    "international", "expatriate", "insurance", "global",
    "medical", "care", "wellbeing", "wellbing", "member",
    "group", "network", "provider", "hospital", "clinic",
}

_PROPER_NOUN_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_PROPER_NOUNS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ── 라틴계 언어 고유 문자 ──────────────────────────────────────
# 전용 문자: 존재만 해도 해당 언어로 확정 (지원 언어 내 다른 언어에 없음)
_DE_EXCLUSIVE = re.compile(r'[ß]')           # ß 는 독일어 전용
_ES_EXCLUSIVE = re.compile(r'[ñÑ]')          # ñ 는 스페인어 전용
_FR_EXCLUSIVE = re.compile(r'[œæŒÆ]')        # œ/æ 는 프랑스어 전용

# 보조 문자: 언어별 전용 악센트 (겹치지 않도록 구분)
# - de: 움라우트 (ä/ö/ü) — fr/es에 없음
# - es: 예각 악센트 모음 (á/í/ó/ú) + 역물음/역느낌표 — fr에 없음
# - fr: 중간/꺾임 악센트 (é/è/ê/à/â/î/ô/ù/û) + ç — es에 없거나 드뭄
_ACCENT_PATTERNS: dict[str, re.Pattern] = {
    "de": re.compile(r'[ÄÖÜäöü]'),
    "es": re.compile(r'[áíóúÁÍÓÚ¡¿]'),
    "fr": re.compile(r'[éàâèêîïôùûçÉÀÂÈÊÎÏÔÙÛÇ]'),
}

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
    2단계: 라틴 전용 문자 → de/es/fr 확정 판별
    3단계: langdetect → 라틴계 언어 판별
    4단계: LLM fallback → 혼합/애매한 경우
    """
    if not text or not text.strip():
        return "en"

    # ── 0단계: 고유명사 제거 후 분석용 텍스트 생성 ────────────
    clean = _PROPER_NOUN_RE.sub(" ", text).strip()
    if not clean:
        clean = text  # 전부 제거된 경우 원문 사용

    # ── 1단계: 스크립트 비율 감지 (ko/ja/zh) ─────────────────
    script_lang = _detect_by_script(clean)
    if script_lang:
        return script_lang

    # ── 2단계: 라틴 전용 문자 감지 (de/es/fr) ────────────────
    latin_lang = _detect_by_latin_exclusive(clean)
    if latin_lang:
        return latin_lang

    # ── 2.5단계: 악센트 문자 점수 감지 ───────────────────────
    accent_lang = _detect_by_accent_score(clean)
    if accent_lang:
        return accent_lang

    # ── 3단계: langdetect ─────────────────────────────────────
    try:
        results = detect_langs(clean)
        top  = results[0]
        lang = LANG_NORMALIZE.get(top.lang, top.lang)
        prob = top.prob

        if lang in SUPPORTED_LANGS and prob >= CONFIDENCE_THRESHOLD:
            return lang

        # 중간 신뢰도(0.55~0.85): 악센트 문자로 보완 확인
        if lang in SUPPORTED_LANGS and prob >= CONFIDENCE_THRESHOLD_LOW:
            accent_lang = _confirm_by_accent(clean, lang)
            if accent_lang:
                return accent_lang

    except Exception:
        pass

    # ── 4단계: LLM fallback ───────────────────────────────────
    return _llm_detect(text)  # 원문 그대로 전달


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _detect_by_latin_exclusive(text: str) -> str | None:
    """
    라틴계 언어 전용 문자로 de/es/fr을 확정 감지한다.

    - ß → de (독일어 전용)
    - ñ → es (스페인어 전용)
    - œ/æ → fr (지원 언어 내 프랑스어 전용)
    """
    if _DE_EXCLUSIVE.search(text):
        return "de"
    if _ES_EXCLUSIVE.search(text):
        return "es"
    if _FR_EXCLUSIVE.search(text):
        return "fr"
    return None


def _detect_by_accent_score(text: str) -> str | None:
    """
    악센트 문자 점수로 de/es/fr을 감지한다.

    각 언어 전용 악센트 문자 수를 세어 한 언어가 명확히 우세하면 반환.
    동점이거나 0이면 None 반환 (langdetect로 위임).
    """
    scores = {lang: len(pat.findall(text)) for lang, pat in _ACCENT_PATTERNS.items()}
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score == 0:
        return None

    others = [v for k, v in scores.items() if k != best]
    second = max(others) if others else 0

    if best_score > second:
        return best

    return None


def _confirm_by_accent(text: str, candidate: str) -> str | None:
    """
    langdetect 중간 신뢰도 결과를 악센트 문자 점수로 보완 확인한다.

    candidate 언어의 악센트 점수가 다른 언어보다 우세하면 candidate를 확정.
    영어(en)는 악센트 문자가 없으므로 보완 없이 None 반환.
    """
    if candidate not in _ACCENT_PATTERNS:
        return None

    scores = {lang: len(pat.findall(text)) for lang, pat in _ACCENT_PATTERNS.items()}
    candidate_score = scores.get(candidate, 0)
    if candidate_score == 0:
        return None

    other_max = max(v for k, v in scores.items() if k != candidate)
    if candidate_score > other_max:
        return candidate

    return None


def _detect_by_script(text: str) -> str | None:
    """
    Unicode 스크립트 블록 비율로 비라틴계 언어를 감지한다.

    - 한글 비율 >= SCRIPT_THRESHOLD → ko
    - 히라가나/가타카나가 1자 이상 + (가나+CJK) 비율 >= SCRIPT_THRESHOLD → ja
      (일본어는 한자를 많이 써도 가나가 섞이므로 가나 존재 여부로 ja/zh 구분)
    - 가나 없이 CJK 비율 >= SCRIPT_THRESHOLD → zh
    """
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return None

    total = len(alpha)

    hangul = sum(1 for ch in alpha if "가" <= ch <= "힣")
    kana   = sum(1 for ch in alpha if "぀" <= ch <= "ヿ")  # 히라가나+가타카나
    cjk    = sum(1 for ch in alpha if "一" <= ch <= "鿿")  # 한자(CJK)

    if hangul / total >= SCRIPT_THRESHOLD:
        return "ko"

    # 가나가 존재하면 일본어 (한자가 많아도 가나는 일본어 전용)
    if kana > 0 and (kana + cjk) / total >= SCRIPT_THRESHOLD:
        return "ja"

    # 가나 없이 CJK만 있으면 중국어
    if cjk / total >= SCRIPT_THRESHOLD:
        return "zh"

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
