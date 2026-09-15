from pathlib import Path
import pandas as pd


INPUT_PATH = Path("notebooks/amazonhelp_golden_set_raw.csv")
OUTPUT_PATH = Path("evaluation/intent_annotation.csv")


df = pd.read_csv(INPUT_PATH)


annotation_df = df[
    [
        "tweet_id",
        "text",
        "conversation_length",
        "message_length_group",
        "conversation_length_group",
    ]
].copy()


annotation_df["intent"] = ""


annotation_df.to_csv(OUTPUT_PATH, index=False)


print("Intent annotation file created successfully.")
print(f"Output: {OUTPUT_PATH}")
print(f"Examples to label: {len(annotation_df)}")

print("\nAllowed intents:")
for intent in [
    "delivery_tracking",
    "order_management",
    "returns_refunds",
    "payment_billing",
    "product_information",
    "technical_support",
    "account_subscription",
    "seller_marketplace",
    "other",
]:
    print(f"- {intent}")