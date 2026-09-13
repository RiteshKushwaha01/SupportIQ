from src.escalation import (
    detect_risk,
    calculate_confidence,
    evaluate_escalation,
)


# ---------------------------------------------------------
# Test 1: Normal customer question
# ---------------------------------------------------------

query = "Where is my package?"

retrieved_results = [
    {"similarity": 0.8365},
    {"similarity": 0.8120},
    {"similarity": 0.7950},
]

result = evaluate_escalation(
    query,
    retrieved_results,
)

print("\n" + "=" * 60)
print("TEST 1: NORMAL QUERY")
print("=" * 60)

print("Query:", query)
print("Risk:", result["risk"])
print("Confidence:", result["confidence"])
print("Escalate:", result["escalate"])
print("Reason:", result["reason"])


# ---------------------------------------------------------
# Test 2: Financial issue
# ---------------------------------------------------------

query = "I was charged twice for my order. I want a refund."

result = evaluate_escalation(
    query,
    retrieved_results,
)

print("\n" + "=" * 60)
print("TEST 2: FINANCIAL QUERY")
print("=" * 60)

print("Query:", query)
print("Risk:", result["risk"])
print("Confidence:", result["confidence"])
print("Escalate:", result["escalate"])
print("Reason:", result["reason"])


# ---------------------------------------------------------
# Test 3: High-risk issue
# ---------------------------------------------------------

query = "Someone hacked my account and made an unauthorized payment."

result = evaluate_escalation(
    query,
    retrieved_results,
)

print("\n" + "=" * 60)
print("TEST 3: HIGH-RISK QUERY")
print("=" * 60)

print("Query:", query)
print("Risk:", result["risk"])
print("Confidence:", result["confidence"])
print("Escalate:", result["escalate"])
print("Reason:", result["reason"])


# ---------------------------------------------------------
# Test 4: Low retrieval confidence
# ---------------------------------------------------------

query = "Can you help me with this issue?"

low_confidence_results = [
    {"similarity": 0.55},
    {"similarity": 0.52},
    {"similarity": 0.49},
]

result = evaluate_escalation(
    query,
    low_confidence_results,
)

print("\n" + "=" * 60)
print("TEST 4: LOW CONFIDENCE")
print("=" * 60)

print("Query:", query)
print("Risk:", result["risk"])
print("Confidence:", result["confidence"])
print("Escalate:", result["escalate"])
print("Reason:", result["reason"])

print("\n" + "=" * 60)
print("ESCALATION TEST COMPLETE")
print("=" * 60)