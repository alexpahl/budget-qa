"""
app.py - Streamlit dashboard for the budget Q&A tool.

Upload a bank CSV, categorize it with Claude (via categorize.py), explore
spending by category and month, and correct any category the AI got wrong.

Run from the project folder with the venv active:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from categorize import (
    CATEGORIES,
    categorize,
    load_cache,
    load_transactions,
    save_cache,
)
from qa import answer_question

st.set_page_config(page_title="Budget Q&A", page_icon="💰", layout="wide")
st.title("Budget Q&A")

# ---------------------------------------------------------------------------
# Sidebar: upload + load
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your data")
    uploaded = st.file_uploader("Bank CSV export", type=["csv"])
    run = st.button("Load & categorize", type="primary", disabled=uploaded is None)
    st.caption(
        "Privacy: your file is processed on this computer. Only merchant "
        "descriptions are sent to Claude, never amounts, dates, or account numbers."
    )

if run and uploaded is not None:
    try:
        with st.spinner("Loading and categorizing (cached merchants are free)..."):
            data = load_transactions(uploaded)
            data = categorize(data)
        st.session_state["df"] = data
        st.session_state["editor_version"] = 0
    except SystemExit as e:  # load_transactions exits on column mismatches
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if "df" not in st.session_state:
    st.info("Upload your bank CSV in the sidebar, then click **Load & categorize**.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    picked = st.date_input(
        "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    if isinstance(picked, tuple) and len(picked) == 2:
        start, end = picked
    else:  # user is mid-selection (only picked one date)
        start, end = min_d, max_d

    exclude_transfers = st.checkbox(
        "Exclude transfers from totals",
        value=True,
        help="Moving money between your own accounts isn't spending or income.",
    )

in_range = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
view = df[in_range]
totals_view = view[view["category"] != "Transfers"] if exclude_transfers else view

# ---------------------------------------------------------------------------
# Summary numbers
# ---------------------------------------------------------------------------
money_in = totals_view.loc[totals_view["amount"] > 0, "amount"].sum()
money_out = -totals_view.loc[totals_view["amount"] < 0, "amount"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Money in", f"${money_in:,.2f}")
c2.metric("Money out", f"${money_out:,.2f}")
c3.metric("Net", f"${money_in - money_out:,.2f}")
c4.metric("Transactions", f"{len(view):,}")

# ---------------------------------------------------------------------------
# Ask your budget
# ---------------------------------------------------------------------------
st.subheader("Ask your budget")
st.caption(
    "Claude turns your question into a query and pandas does the math. "
    "Questions search all loaded transactions, regardless of the date filter. "
    "Your question and the computed result are sent to Claude; raw rows are not."
)

if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

with st.form("qa_form", clear_on_submit=True):
    question = st.text_input(
        "Question", placeholder="How much did I spend on dining in September?"
    )
    asked = st.form_submit_button("Ask")

if asked and question.strip():
    with st.spinner("Thinking..."):
        try:
            outcome = answer_question(df, question.strip())
        except Exception as e:
            outcome = {
                "answer": f"Something went wrong: {e}",
                "plan": None,
                "result": None,
            }
    st.session_state["qa_history"].insert(0, {"question": question.strip(), **outcome})

for item in st.session_state["qa_history"][:5]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        # Escape $ so Streamlit doesn't treat "$10 and $20" as a math formula
        st.markdown(item["answer"].replace("$", r"\$"))
        if item.get("plan"):
            with st.expander("How this was calculated"):
                st.json({"query": item["plan"], "result": item["result"]})

# ---------------------------------------------------------------------------
# Charts (spending only)
# ---------------------------------------------------------------------------
spend = totals_view[totals_view["amount"] < 0].copy()
spend["spent"] = -spend["amount"]

left, right = st.columns(2)
with left:
    st.subheader("Spending by category")
    if spend.empty:
        st.write("No spending in this range.")
    else:
        by_cat = spend.groupby("category")["spent"].sum().sort_values(ascending=False)
        st.bar_chart(by_cat)
with right:
    st.subheader("Spending by month")
    if spend.empty:
        st.write("No spending in this range.")
    else:
        by_month = spend.groupby(spend["date"].dt.to_period("M").astype(str))[
            "spent"
        ].sum()
        st.bar_chart(by_month)

# ---------------------------------------------------------------------------
# Transactions table with category correction
# ---------------------------------------------------------------------------
st.subheader("Transactions")
st.caption(
    "Change a category in the table, then click **Save corrections**. "
    "Corrections apply to every transaction from that merchant and are "
    "remembered for future uploads."
)

cat_filter = st.multiselect("Show only these categories", CATEGORIES)
table_view = view[view["category"].isin(cat_filter)] if cat_filter else view

shown_cols = ["date", "description", "amount", "category"]
version = st.session_state.get("editor_version", 0)
edited = st.data_editor(
    table_view[shown_cols],
    column_config={
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
    },
    disabled=["date", "description", "amount"],
    hide_index=True,
    use_container_width=True,
    key=f"editor_{version}",
)

if st.button("Save corrections"):
    original = table_view.loc[edited.index, "category"]
    changed = edited.index[edited["category"] != original]
    if len(changed) == 0:
        st.info("No changes to save.")
    else:
        cache = load_cache()
        for idx in changed:
            cache[df.loc[idx, "merchant"]] = edited.loc[idx, "category"]
        save_cache(cache)
        df["category"] = df["merchant"].map(cache).fillna("Other")
        st.session_state["df"] = df
        st.session_state["editor_version"] = version + 1
        st.rerun()

st.download_button(
    "Download categorized CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="categorized.csv",
    mime="text/csv",
)