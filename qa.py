"""
qa.py - Ask questions about your spending in plain English.

How it works:
  1. Claude gets your question plus a short description of your data (date
     range, category list). It never sees your transactions. It replies with
     a structured query: what to measure, which categories, which dates...
  2. pandas runs that query on your data. All the math happens here.
  3. Claude turns the computed result into a plain-English answer.

Privacy: your question, the query, and the computed result (totals, counts,
merchant names for "top merchants" style questions) are sent to Claude.
Raw rows, account numbers, and balances are not.
"""

import json
from datetime import date

import anthropic
import pandas as pd

from categorize import CATEGORIES, MODEL  # also loads your .env

MAX_GROUPS = 25

QUERY_TOOL = {
    "name": "run_query",
    "description": (
        "Run a query over the user's categorized bank transactions and return "
        "totals, counts, or averages, optionally grouped."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["total", "count", "average"],
                "description": "total = sum of amounts, count = number of "
                "transactions, average = average amount per transaction.",
            },
            "direction": {
                "type": "string",
                "enum": ["spending", "income", "both"],
                "description": "spending = money out, income = money in, "
                "both = net of everything. Default to spending for words "
                "like spend, cost, paid, bought.",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": CATEGORIES},
                "description": "Only include these categories. Omit for all.",
            },
            "merchant_contains": {
                "type": "string",
                "description": "Only include transactions whose description "
                "contains this text (case-insensitive), e.g. 'starbucks'.",
            },
            "start_date": {
                "type": "string",
                "description": "Inclusive start date, YYYY-MM-DD.",
            },
            "end_date": {
                "type": "string",
                "description": "Inclusive end date, YYYY-MM-DD.",
            },
            "group_by": {
                "type": "string",
                "enum": ["none", "category", "month", "merchant"],
                "description": "Break the result down by this. Use 'merchant' "
                "for 'where do I spend the most' style questions.",
            },
            "top_n": {
                "type": "integer",
                "description": f"For category/merchant grouping, how many "
                f"groups to return (default 10, max {MAX_GROUPS}).",
            },
            "exclude_transfers": {
                "type": "boolean",
                "description": "Skip transfers between the user's own "
                "accounts. Defaults to true; set false only if the user "
                "asks about transfers.",
            },
        },
        "required": ["metric", "direction"],
    },
}


def _text(response) -> str:
    """Join the text blocks of a reply (it may also contain thinking blocks)."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _data_range(df: pd.DataFrame) -> tuple[str, str]:
    return df["date"].min().date().isoformat(), df["date"].max().date().isoformat()


def build_system_prompt(df: pd.DataFrame) -> str:
    first, last = _data_range(df)
    return (
        "You answer questions about a person's bank transactions by calling "
        "the run_query tool. You never see the transactions themselves.\n\n"
        f"Today's date: {date.today().isoformat()}\n"
        f"The data covers {first} to {last} ({len(df)} transactions).\n"
        "Categories: " + ", ".join(CATEGORIES) + "\n\n"
        "Rules:\n"
        "- For any question about spending, income, or transactions, call "
        "run_query. Never state numbers yourself.\n"
        "- Resolve relative dates ('last month', 'this year') using today's "
        "date into start_date and end_date.\n"
        "- Map everyday wording to categories ('eating out' -> Dining, 'gas' "
        "-> Fuel). Use merchant_contains for a specific store or company.\n"
        "- If the question can't be answered from date, description, amount "
        "and category, reply in plain text without calling the tool and say "
        "what you can answer instead."
    )


def plan_query(client: anthropic.Anthropic, df: pd.DataFrame, question: str):
    """Ask Claude for a query plan. Returns (plan, message); plan is None if
    Claude declined to run a query, in which case message explains why."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=build_system_prompt(df),
        tools=[QUERY_TOOL],
        messages=[{"role": "user", "content": question}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return dict(block.input), ""
    return None, _text(response) or "I couldn't turn that into a query. Try rephrasing."


def _parse_date(value):
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Run a query plan with pandas. No AI involved: this is where the math is."""
    metric = plan.get("metric", "total")
    if metric not in ("total", "count", "average"):
        metric = "total"
    direction = plan.get("direction", "spending")
    if direction not in ("spending", "income", "both"):
        direction = "spending"
    categories = [c for c in (plan.get("categories") or []) if c in CATEGORIES]
    group_by = plan.get("group_by", "none")

    work = df

    # Transfers between your own accounts aren't spending or income
    if plan.get("exclude_transfers", True) and "Transfers" not in categories:
        work = work[work["category"] != "Transfers"]
    if categories:
        work = work[work["category"].isin(categories)]

    start, end = _parse_date(plan.get("start_date")), _parse_date(plan.get("end_date"))
    if start is not None:
        work = work[work["date"] >= start]
    if end is not None:
        work = work[work["date"] < end + pd.Timedelta(days=1)]

    needle = (plan.get("merchant_contains") or "").strip()
    if needle:
        hit = work["description"].str.contains(
            needle, case=False, regex=False
        ) | work["merchant"].str.contains(needle, case=False, regex=False)
        work = work[hit]

    if direction == "spending":
        work = work[work["amount"] < 0].assign(value=lambda d: -d["amount"])
    elif direction == "income":
        work = work[work["amount"] > 0].assign(value=lambda d: d["amount"])
    else:
        work = work.assign(value=lambda d: d["amount"])

    def aggregate(series: pd.Series):
        if metric == "count":
            return int(series.count())
        if len(series) == 0:
            return 0.0
        return round(float(series.mean() if metric == "average" else series.sum()), 2)

    groups = []
    if group_by in ("category", "month", "merchant") and not work.empty:
        keys = {
            "category": work["category"],
            "month": work["date"].dt.to_period("M").astype(str),
            "merchant": work["merchant"],
        }[group_by]
        grouped = work["value"].groupby(keys)
        if metric == "count":
            out = grouped.count()
        elif metric == "average":
            out = grouped.mean()
        else:
            out = grouped.sum()

        if group_by == "month":
            out = out.sort_index().tail(MAX_GROUPS)  # most recent months
        else:
            top_n = max(1, min(int(plan.get("top_n") or 10), MAX_GROUPS))
            out = out.sort_values(ascending=False).head(top_n)

        groups = [
            {"group": str(k), "value": int(v) if metric == "count" else round(float(v), 2)}
            for k, v in out.items()
        ]

    return {
        "metric": metric,
        "direction": direction,
        "transaction_count": int(len(work)),
        "overall": aggregate(work["value"]),
        "groups": groups,
        "filters": {
            "categories": categories or "all",
            "merchant_contains": needle or None,
            "start_date": start.date().isoformat() if start is not None else None,
            "end_date": end.date().isoformat() if end is not None else None,
            "transfers_excluded": bool(
                plan.get("exclude_transfers", True) and "Transfers" not in categories
            ),
        },
    }


def phrase_answer(client: anthropic.Anthropic, df, question, plan, result) -> str:
    first, last = _data_range(df)
    prompt = (
        f"A person asked: {question}\n\n"
        f"A query was run on their transactions.\n"
        f"Query: {json.dumps(plan)}\n"
        f"Result: {json.dumps(result)}\n"
        f"Their data covers {first} to {last}.\n\n"
        "Answer the question in 1-3 plain sentences using ONLY numbers that "
        "appear in the result. Mention the date range or category used when "
        "it helps. Show money in dollars with two decimals. If "
        "transaction_count is 0, say nothing matched and mention the data "
        "range. Do not invent numbers or give financial advice."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text(response) or "I ran the query but couldn't phrase an answer."


def answer_question(df: pd.DataFrame, question: str) -> dict:
    """Full pipeline. Returns {'answer': str, 'plan': dict|None, 'result': dict|None}."""
    client = anthropic.Anthropic()
    plan, message = plan_query(client, df, question)
    if plan is None:
        return {"answer": message, "plan": None, "result": None}
    result = execute_plan(df, plan)
    return {
        "answer": phrase_answer(client, df, question, plan, result),
        "plan": plan,
        "result": result,
    }