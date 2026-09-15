from pathlib import Path
import pandas as pd


INPUT_PATH = Path("evaluation/intent_annotation.csv")
OUTPUT_PATH = Path("evaluation/intent_annotation.csv")

INTENTS = [
    "delivery_tracking",
    "order_management",
    "returns_refunds",
    "payment_billing",
    "product_information",
    "technical_support",
    "account_subscription",
    "seller_marketplace",
    "other",
]

BATCH_SIZE = 20


df = pd.read_csv(INPUT_PATH)

df["intent"] = df["intent"].fillna("").astype("string")


# Find unlabeled examples
unlabeled = df[
    df["intent"].isna() | (df["intent"].str.strip() == "")
]


print(f"Total examples: {len(df)}")
print(f"Already labeled: {len(df) - len(unlabeled)}")
print(f"Remaining: {len(unlabeled)}")


if unlabeled.empty:
    print("\nAll examples are already labeled.")
    raise SystemExit(0)


# Take the next batch
batch = unlabeled.head(BATCH_SIZE)

print("\n" + "=" * 90)
print(f"NEXT BATCH — {len(batch)} examples")
print("=" * 90)

for number, (index, row) in enumerate(batch.iterrows(), start=1):

    print("\n" + "-" * 90)
    print(f"Example {number}/{len(batch)}   |   Golden-set row: {index}")
    print("-" * 90)

    print(row["text"])


print("\n" + "=" * 90)
print("INTENTS")
print("=" * 90)

for number, intent in enumerate(INTENTS, start=1):
    print(f"{number}. {intent}")


print("\nEnter exactly one number for each example.")
print("Example: 1 4 9 6 1 2 3 9 7 4 ...")
print("Enter q to quit without saving this batch.")


while True:

    choices = input(
        f"\nEnter {len(batch)} labels separated by spaces: "
    ).strip()

    if choices.lower() == "q":
        print("\nNo changes made to this batch.")
        raise SystemExit(0)

    values = choices.split()

    if len(values) != len(batch):
        print(
            f"Error: expected {len(batch)} labels, "
            f"but received {len(values)}."
        )
        continue

    if not all(
        value.isdigit() and 1 <= int(value) <= len(INTENTS)
        for value in values
    ):
        print("Error: every label must be a number from 1 to 9.")
        continue

    break


# Save labels
for (index, _), value in zip(batch.iterrows(), values):
    df.at[index, "intent"] = INTENTS[int(value) - 1]


df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 90)
print("BATCH SAVED SUCCESSFULLY")
print("=" * 90)

print(f"Labeled in this batch: {len(batch)}")
print(f"Total labeled: {df['intent'].notna().sum()}")
print(f"Remaining: {len(df) - df['intent'].notna().sum()}")

print("\nRun the script again for the next batch.")