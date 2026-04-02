import uuid
from datetime import date, datetime
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def load_transactions() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{API_URL}/transactions", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        st.error(f"Failed to load transactions from API: {e}")
        return []


def delete_transaction(tx_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_URL}/transactions/{tx_id}", timeout=5)
        if resp.status_code in (200, 204):
            return True
        st.error(f"Failed to delete transaction {tx_id}: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        st.error(f"Error while deleting transaction {tx_id}: {e}")
        return False


def replace_all(transactions: List[Dict[str, Any]]) -> bool:
    try:
        resp = requests.put(f"{API_URL}/transactions/bulk", json=transactions, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to save transactions to API: {e}")
        return False


st.set_page_config(page_title="Budget DB Admin", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 1rem;
    }
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 40%, #020617 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid transparent;
        padding: 0.45rem 1.3rem;
        font-weight: 500;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: #ffffff;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }
    @media (max-width: 768px) {
        .main > div {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Budget Database Admin")

st.sidebar.header("Filters")

raw_data = load_transactions()
df = pd.DataFrame(raw_data)

if df.empty:
    st.info("No transactions found in the database.")
else:
    # make sure amount and date behave for filtering
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # filter controls live in the sidebar
    min_date = df["date"].min().date() if not df["date"].isna().all() else date.today()
    max_date = df["date"].max().date() if not df["date"].isna().all() else date.today()

    start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    tx_types = st.sidebar.multiselect(
        "Types",
        options=sorted(df["type"].dropna().unique().tolist()),
        default=sorted(df["type"].dropna().unique().tolist()),
    )

    category_filter = st.sidebar.text_input("Category contains")
    note_filter = st.sidebar.text_input("Note contains")

    # narrow down the dataframe
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    if tx_types:
        mask &= df["type"].isin(tx_types)
    if category_filter:
        mask &= df["category"].str.contains(category_filter, case=False, na=False)
    if note_filter:
        mask &= df["note"].str.contains(note_filter, case=False, na=False)

    df_filtered = df.loc[mask].copy()
    # small numbers in the table are easier than long uuids
    df_filtered = df_filtered.sort_values(by="date", ascending=False).reset_index(drop=True)
    df_filtered["short_id"] = df_filtered.index + 1

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total records", len(df))
    with col_b:
        st.metric("Filtered records", len(df_filtered))
    with col_c:
        total_income = df_filtered.loc[df_filtered["type"] == "income", "amount"].sum()
        st.metric("Filtered income", f"{total_income:,.2f}")
    with col_d:
        total_expense = df_filtered.loc[df_filtered["type"] == "expense", "amount"].sum()
        st.metric("Filtered expense", f"{total_expense:,.2f}")

    st.subheader("Filtered Transactions")
    display_cols = ["short_id", "date", "type", "amount", "category", "note"]
    existing_cols = [c for c in display_cols if c in df_filtered.columns]
    display_df = df_filtered[existing_cols].rename(columns={"short_id": "ID"})
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Actions")

    # delete one row using the short id from the table
    st.markdown("**Delete single transaction by ID (short number from table)**")
    col_id, col_btn = st.columns([3, 1])
    with col_id:
        delete_id = st.text_input("Short ID to delete", placeholder="Enter the small ID number from the table above")
    with col_btn:
        if st.button("Delete by ID", type="primary", use_container_width=True):
            if delete_id.strip():
                try:
                    short_id_int = int(delete_id.strip())
                except ValueError:
                    st.warning("Short ID must be a number.")
                else:
                    id_map = dict(zip(df_filtered["short_id"].astype(int), df_filtered["id"].astype(str)))
                    real_id = id_map.get(short_id_int)
                    if not real_id:
                        st.warning("No transaction found with that short ID in the current filter.")
                    else:
                        ok = delete_transaction(real_id)
                        if ok:
                            st.success(f"Transaction with ID {short_id_int} deleted.")
                            st.experimental_rerun()
            else:
                st.warning("Please enter a short ID.")

    st.markdown("---")

    # wipe several at once from the multiselect
    st.markdown("**Bulk delete (selected short IDs)**")
    short_id_options = df_filtered["short_id"].astype(int).tolist()
    selected_short_ids = st.multiselect("Select short IDs to delete", options=short_id_options)

    if st.button("Delete selected", type="secondary"):
        if not selected_short_ids:
            st.warning("No short IDs selected.")
        else:
            id_map = dict(zip(df_filtered["short_id"].astype(int), df_filtered["id"].astype(str)))
            deleted_count = 0
            for sid in selected_short_ids:
                real_id = id_map.get(int(sid))
                if real_id and delete_transaction(real_id):
                    deleted_count += 1
            if deleted_count:
                st.success(f"Deleted {deleted_count} transaction(s).")
                st.experimental_rerun()

    st.markdown("---")

    # nuke everything (scary)
    st.markdown("**Danger zone**")
    clear = st.checkbox("I understand this will permanently remove all records.")
    if st.button("Clear entire database", type="primary", disabled=not clear):
        if clear:
            ok = replace_all([])
            if ok:
                st.success("All transactions cleared from the database.")
                st.experimental_rerun()

