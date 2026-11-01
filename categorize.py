"""
categorize.py - Stages 1 and 2 of the budget Q&A tool.

Stage 1: load a bank CSV export and clean it up with pandas.
Stage 2: categorize each transaction with the Claude API.

Only merchant descriptions are sent to the API (no amounts or dates).
Results are cached by merchant, so you never pay to categorize the same
merchant twice.

Usage (from the project folder, with the venv active):
    python categorize.py my_transactions.csv
"""

import json
import re
import sys
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from your .env file

# ---------------------------------------------------------------------------
# CONFIG - edit these to match your bank's CSV export
# ---------------------------------------------------------------------------
DATE_COL = "Post Date"
DESC_COL = "Description"
DEBIT_COL = "Debit"  # money out
CREDIT_COL = "Credit"  # money in

# Cheapest current model; fine for categorizing. Check docs.claude.com for
# current model names if this one ever stops working.
MODEL = "claude-haiku-5-5"

CATEGORIES = [
    "Groceries",
    "Dining",
    "Fuel",
    "Transportation",
    "Auto (maintenance/insurance/payments)",
    "Housing (rent/mortgage/utilities)",
    "Subscriptions",
    "Shopping",
    "Health",
    "Entertainment",
    "Travel",
    "Income",
    "Transfers",
    "Fees",
    "Other",
]

CACHE_FILE = Path("categories_cache.json")
OUTPUT_FILE = Path("categorized.csv")
BATCH_SIZE = 40


# ---------------------------------------------------------------------------
# Stage 1: load and clean
# ---------------------------------------------------------------------------
def load_transactions(path: str) -> pd.DataFrame:
    # sep=None auto-detects whether the file uses commas or tabs
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    needed_cols = (DATE_COL, DESC_COL, DEBIT_COL, CREDIT_COL)
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"Column(s) not found in CSV: {missing}\n"
            f"Columns in your file: {list(df.columns)}\n"
            "Update the column names in the CONFIG section at the top of this script."
        )

    def to_number(series: pd.Series) -> pd.Series:
        # Handles "$1,234.56" and "(45.00)" style values; blanks become 0
        cleaned = (
            series.astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0)

    # Credits are money in (positive), debits are money out (negative).
    # abs() makes this work whether your bank lists debits as 45 or -45.
    df["amount"] = to_number(df[CREDIT_COL]).abs() - to_number(df[DEBIT_COL]).abs()

    # Only the columns we need; the account number is dropped on purpose.
    df = df.rename(columns={DATE_COL: "date", DESC_COL: "description"})[
        ["date", "description", "amount"]
    ]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "amount"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Note: dropped {dropped} row(s) with an unreadable date or amount.")

    df["description"] = df["description"].fillna("").astype(str)
    df["merchant"] = df["description"].apply(normalize_merchant)
    return df


def normalize_merchant(description: str) -> str:
    """Turn 'STARBUCKS #1234 MINNEAPOLIS MN' into 'STARBUCKS MINNEAPOLIS MN'."""
    s = description.upper()
    s = re.sub(r"[#*]\s*\w+", " ", s)  # store numbers / reference codes
    s = re.sub(r"\d+", " ", s)  # stray digits
    s = re.sub(r"[^A-Z& ]", " ", s)  # punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(s.split()[:4]) or "UNKNOWN"


# ---------------------------------------------------------------------------
# Stage 2: categorize with Claude
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def categorize_batch(client: anthropic.Anthropic, merchants: list[str]) -> dict:
    prompt = (
        "Categorize each merchant/transaction description into exactly one of "
        "these categories:\n"
        + "\n".join(f"- {c}" for c in CATEGORIES)
        + "\n\nReply with ONLY a JSON object mapping each description to its "
        "category, using the descriptions exactly as given. No other text.\n\n"
        "Descriptions:\n"
        + "\n".join(f"- {m}" for m in merchants)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    # The response can include non-text blocks (e.g. thinking) before the answer,
    # so collect only the text blocks instead of assuming the first one is text.
    text = "".join(b.text for b in response.content if b.type == "text")

    # Be forgiving if the model wraps the JSON in extra text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        print(
            "Warning: no JSON found in a response batch "
            f"(stop_reason={response.stop_reason}). Not cached; re-run to retry."
        )
        print("Reply started with:", repr(text[:200]))
        return {}

    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        print(
            "Warning: invalid JSON in a response batch "
            f"(stop_reason={response.stop_reason}). Not cached; re-run to retry."
        )
        return {}

    # Validate: anything missing or not in our list becomes 'Other'
    return {
        m: raw.get(m) if raw.get(m) in CATEGORIES else "Other" for m in merchants
    }


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    needed = sorted(set(df["merchant"]) - set(cache))

    if needed:
        print(f"Categorizing {len(needed)} new merchant(s) with Claude...")
        client = anthropic.Anthropic()
        for i in range(0, len(needed), BATCH_SIZE):
            batch = needed[i : i + BATCH_SIZE]
            cache.update(categorize_batch(client, batch))
            save_cache(cache)  # save as we go so progress isn't lost
    else:
        print("All merchants already cached - no API calls needed.")

    df["category"] = df["merchant"].map(cache).fillna("Other")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python categorize.py your_transactions.csv")

    df = load_transactions(sys.argv[1])
    print(f"Loaded {len(df)} transactions.")

    df = categorize(df)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 40)
    print("\nFirst 20 transactions:")
    print(df[["date", "description", "amount", "category"]].head(20).to_string())

    print("\nSpending by category (negative = money out):")
    print(df.groupby("category")["amount"].sum().sort_values().round(2).to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved full results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
