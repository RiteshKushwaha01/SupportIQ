from src.intent_model import load_intent_classifier


classifier = load_intent_classifier()


test_messages = [
    "Where is my package?",
    "My order has not arrived yet.",
    "I want to return this product.",
    "I was charged for Amazon Prime.",
    "Alexa is not working.",
    "Someone hacked my account.",
]


for message in test_messages:

    result = classifier.predict(message)

    print()
    print("Message:", message)
    print("Intent:", result["intent"])
    print("Semantic Score:", round(result["semantic_score"], 4))