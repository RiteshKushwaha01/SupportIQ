"""
Risk and confidence based escalation logic
for the SupportIQ customer support agent.
"""


# ---------------------------------------------------------
# Risk categories
# ---------------------------------------------------------

HIGH_RISK_KEYWORDS = {
    "fraud",
    "scam",
    "hacked",
    "stolen",
    "unauthorized",
    "identity theft",
    "account compromised",
}

FINANCIAL_KEYWORDS = {
    "charged twice",
    "double charged",
    "refund",
    "money back",
    "payment",
    "billing",
    "charge",
    "credit card",
    "debit card",
}

ACCOUNT_KEYWORDS = {
    "password",
    "login",
    "locked out",
    "account access",
    "verification",
    "security",
}


def detect_risk(query):
    """
    Detect potential risk categories in a customer query.

    Returns:
        {
            "risk_level": str,
            "risk_categories": list
        }
    """

    text = query.lower()

    categories = []

    # High-risk issues
    if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
        categories.append("high_risk")

    # Financial issues
    if any(keyword in text for keyword in FINANCIAL_KEYWORDS):
        categories.append("financial")

    # Account/security issues
    if any(keyword in text for keyword in ACCOUNT_KEYWORDS):
        categories.append("account_security")

    # Determine overall risk
    if "high_risk" in categories:
        risk_level = "high"

    elif categories:
        risk_level = "medium"

    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "risk_categories": categories,
    }


# ---------------------------------------------------------
# Retrieval confidence
# ---------------------------------------------------------

def calculate_confidence(retrieved_results):
    """
    Calculate retrieval confidence from FAISS similarity.

    Uses the highest similarity score and the average
    similarity of the retrieved results.
    """

    if not retrieved_results:
        return {
            "confidence": 0.0,
            "confidence_level": "low",
        }

    scores = []

    for result in retrieved_results:
        # Our retriever stores the similarity score.
        score = result.get("similarity", 0.0)
        scores.append(float(score))

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    # Weighted confidence:
    # strongest result matters most.
    confidence = (
        0.7 * max_score +
        0.3 * avg_score
    )

    # Provisional threshold.
    # We will calibrate this using the golden set later.
    if confidence >= 0.80:
        confidence_level = "high"

    elif confidence >= 0.70:
        confidence_level = "medium"

    else:
        confidence_level = "low"

    return {
        "confidence": round(confidence, 4),
        "confidence_level": confidence_level,
        "max_similarity": round(max_score, 4),
        "average_similarity": round(avg_score, 4),
    }


# ---------------------------------------------------------
# Escalation decision
# ---------------------------------------------------------

def should_escalate(confidence_info, risk_info):
    """
    Decide whether a customer query should be handled
    automatically or escalated to a human agent.
    """

    risk_level = risk_info["risk_level"]
    risk_categories = risk_info["risk_categories"]
    confidence_level = confidence_info["confidence_level"]

    # -------------------------------------------------
    # 1. High-risk issues always go to a human.
    # -------------------------------------------------

    if risk_level == "high":
        return True, "high_risk"

    # -------------------------------------------------
    # 2. Financial issues require human review.
    # -------------------------------------------------

    if "financial" in risk_categories:
        return True, "financial_issue"

    # -------------------------------------------------
    # 3. Account/security issues with anything below
    #    high confidence go to a human.
    # -------------------------------------------------

    if "account_security" in risk_categories:
        if confidence_level != "high":
            return True, "account_security_low_confidence"

    # -------------------------------------------------
    # 4. Low retrieval confidence means insufficient
    #    evidence to safely generate an answer.
    # -------------------------------------------------

    if confidence_level == "low":
        return True, "low_retrieval_confidence"

    # -------------------------------------------------
    # 5. Otherwise AI can handle the request.
    # -------------------------------------------------

    return False, "safe_for_ai"

# ---------------------------------------------------------
# Complete decision
# ---------------------------------------------------------

def evaluate_escalation(query, retrieved_results):
    """
    Perform the complete risk + confidence evaluation.
    """

    risk_info = detect_risk(query)

    confidence_info = calculate_confidence(
        retrieved_results
    )

    escalate, reason = should_escalate(
        confidence_info,
        risk_info,
    )

    return {
        "risk": risk_info,
        "confidence": confidence_info,
        "escalate": escalate,
        "reason": reason,
    }