# PII 차단, RECOMMENDATION_KEYWORDS, COMPARISON_KEYWORDS, check_blocked()
import re

# 특정 보험 추천 요청 — 차단 대상
RECOMMENDATION_KEYWORDS = ["추천해줘", "골라줘", "어떤 보험이 좋아", "recommend", "which is better"]

# 보험사 비교 요청 — 차단 아님, compare_node로 라우팅
COMPARISON_KEYWORDS = ["비교해줘", "비교해주세요", "compare", "vs", "difference between"]

# 개인정보 패턴
PII_PATTERNS = [
    r"\d{6}-\d{7}",      # 주민등록번호
    r"[A-Z]\d{8}",       # 여권번호
    r"\d{3}-\d{2}-\d{4}",  # SSN (미국)
]


def check_blocked(text: str) -> str | None:
    """
    차단 대상이면 안내 메시지 반환, 아니면 None 반환
    """
    # PII 패턴 감지
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return "개인정보가 포함된 것 같습니다. 개인정보를 입력하지 말아 주세요."

    # 추천 요청 차단
    if any(k in text for k in RECOMMENDATION_KEYWORDS):
        return "특정 보험 추천은 제공하지 않습니다. 각 보험사의 혜택을 비교해드릴 수 있어요."

    return None
