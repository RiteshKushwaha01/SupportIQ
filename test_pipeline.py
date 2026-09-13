from src.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()


queries = [
    "Where is my package?",
    "I was charged twice and I want a refund.",
    "Someone hacked my account and made an unauthorized payment.",
    "Can you help me with this issue?",
]


for query in queries:

    print("\n" + "=" * 70)
    print("CUSTOMER:")
    print(query)

    result = pipeline.generate_response(query)

    print("\nDECISION:")
    print(result["decision"])

    print("ESCALATE:")
    print(result["escalate"])

    print("REASON:")
    print(result["reason"])

    print("RISK:")
    print(result["risk_level"])

    print("RISK CATEGORIES:")
    print(result["risk_categories"])

    print("CONFIDENCE:")
    print(result["confidence"])

    print("CONFIDENCE LEVEL:")
    print(result["confidence_level"])

    if result["answer"]:
        print("\nAI RESPONSE:")
        print(result["answer"])

    else:
        print("\nAI RESPONSE:")
        print("Not generated — escalated to human agent.")