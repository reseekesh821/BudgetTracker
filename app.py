import os
import json
import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
import plotly.express as px
import uuid
import logging
from typing import List, Dict, Any, Optional
import requests
from streamlit import config as st_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "transactions.json")
INCOME_CATEGORIES = ["Salary", "Freelance", "Investment", "Gift", "Bonus", "Other"]
EXPENSE_CATEGORIES = ["Groceries", "Rent", "Utilities", "Dining Out", "Transport", "Entertainment", "Clothing", "Healthcare", "Other"]
ALL_CATEGORIES = sorted(list(set(INCOME_CATEGORIES + EXPENSE_CATEGORIES)))
CURRENCY = "$"
API_URL = "http://127.0.0.1:8000"

os.makedirs(DATA_DIR, exist_ok=True)


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        /*
         * Built-in Streamlit stuff uses the theme from config / settings.
         * The history cards below are my own HTML — I used Field / FieldText so they
         * don't stay bright white when the app is in dark mode.
         */
        :root {
            --accent-primary: #3b82f6;
            --success: #10b981;
            --success-bg: #d1fae5;
            --danger: #ef4444;
            --danger-bg: #fee2e2;
            --radius-md: 12px;
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        section[data-testid="stMain"] .block-container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 1rem;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid color-mix(in oklab, FieldText 14%, transparent);
        }

        .app-header-icon {
            width: 48px;
            height: 48px;
            border-radius: var(--radius-md);
            background: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-md);
        }

        .app-header-title {
            font-size: 1.875rem;
            font-weight: 700;
            margin: 0;
            color: FieldText;
            letter-spacing: -0.025em;
        }

        .app-header-subtitle {
            margin: 0.25rem 0 0 0;
            font-size: 0.95rem;
            color: color-mix(in oklab, FieldText 62%, transparent);
        }

        .statement-date-label {
            font-size: 0.875rem;
            font-weight: 600;
            color: color-mix(in oklab, FieldText 58%, transparent);
            margin: 1.5rem 0 0.5rem 0;
            border-bottom: 1px solid color-mix(in oklab, FieldText 14%, transparent);
            padding-bottom: 0.25rem;
        }

        .statement-card {
            background: Field !important;
            color: FieldText !important;
            border: 1px solid color-mix(in oklab, FieldText 18%, transparent) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 1px 2px color-mix(in oklab, FieldText 10%, transparent) !important;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .statement-card-left {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .statement-card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .statement-card-category {
            font-weight: 600 !important;
            font-size: 1rem !important;
            color: FieldText !important;
        }

        .statement-card-meta {
            font-size: 0.875rem !important;
            color: color-mix(in oklab, FieldText 62%, transparent) !important;
        }

        .statement-card-amount-income {
            font-weight: 700 !important;
            font-size: 1.125rem !important;
            color: var(--success) !important;
        }

        .statement-card-amount-expense {
            font-weight: 700 !important;
            font-size: 1.125rem !important;
            color: FieldText !important;
        }

        .statement-badge {
            font-size: 0.75rem;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            font-weight: 500;
        }

        .statement-badge-income {
            background: color-mix(in oklab, var(--success) 20%, Field) !important;
            color: var(--success) !important;
        }
        .statement-badge-expense {
            background: color-mix(in oklab, var(--danger) 20%, Field) !important;
            color: var(--danger) !important;
        }

        /* backup colors if the browser can't do Field / FieldText */
        @supports not (color: FieldText) {
            .app-header-title { color: #0f172a; }
            .app-header-subtitle { color: #64748b; }
            .statement-date-label { color: #64748b; border-bottom-color: #e2e8f0; }
            .statement-card {
                background: #ffffff !important;
                color: #0f172a !important;
                border-color: #e2e8f0 !important;
            }
            .statement-card-category { color: #0f172a !important; }
            .statement-card-meta { color: #64748b !important; }
            .statement-card-amount-expense { color: #0f172a !important; }
            .statement-badge-income { background: #d1fae5 !important; color: #059669 !important; }
            .statement-badge-expense { background: #fee2e2 !important; color: #dc2626 !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_sidebar_primary_button_styles() -> None:
    """Makes the disabled Add button look tinted instead of that empty gray outline (uses theme colors from config)."""
    primary = st_config.get_option("theme.primaryColor") or "#2563eb"
    surface = st_config.get_option("theme.secondaryBackgroundColor") or "#ffffff"
    text = st_config.get_option("theme.textColor") or "#0f172a"
    st.markdown(
        f"""
        <style>
        :root {{
            --app-theme-primary: {primary};
            --app-theme-surface: {surface};
            --app-theme-text: {text};
        }}
        /* sidebar Add button when it's disabled — still looks like the blue button, just faded */
        section[data-testid="stSidebar"] [data-testid="stButton"] button:disabled {{
            background-color: color-mix(in oklab, var(--app-theme-primary) 40%, var(--app-theme-surface)) !important;
            color: color-mix(in oklab, var(--app-theme-primary) 25%, var(--app-theme-text)) !important;
            border: 1px solid color-mix(in oklab, var(--app-theme-primary) 55%, transparent) !important;
            opacity: 1 !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stButton"] button:disabled:hover {{
            background-color: color-mix(in oklab, var(--app-theme-primary) 44%, var(--app-theme-surface)) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
          <div class="app-header-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <rect x="3" y="5" width="18" height="14" rx="3.5" stroke="white" stroke-width="2" />
              <path d="M16.5 12H20" stroke="white" stroke-width="2" stroke-linecap="round" />
              <circle cx="14" cy="12" r="1.6" fill="white" />
            </svg>
          </div>
          <div>
            <h1 class="app-header-title">Personal Budget Tracker</h1>
            <p class="app-header-subtitle">Track income, expenses, and trends with a clean, responsive dashboard.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_transactions(filepath: str = DATA_FILE) -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/transactions", timeout=5)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            return []

        valid_transactions: List[Dict[str, Any]] = []
        for t in data:
            if not isinstance(t, dict):
                continue
            t_id = t.get("id") or str(uuid.uuid4())
            t_type = t.get("type")
            if t_type not in ["income", "expense"]:
                continue
            try:
                amount = abs(float(t.get("amount", 0.0)))
            except (ValueError, TypeError):
                amount = 0.0
            if amount == 0.0:
                continue
            category = str(t.get("category", "Other")).strip() or "Other"
            date_val = t.get("date")
            if isinstance(date_val, date):
                date_str = date_val.isoformat()
            else:
                try:
                    datetime.strptime(str(date_val), "%Y-%m-%d")
                    date_str = str(date_val)
                except Exception:
                    date_str = datetime.now().date().isoformat()
            note = str(t.get("note", "")).strip()

            valid_transactions.append(
                {
                    "id": t_id,
                    "type": t_type,
                    "amount": amount,
                    "category": category,
                    "date": date_str,
                    "note": note,
                }
            )

        valid_transactions.sort(key=lambda x: x.get("date", "1970-01-01"), reverse=True)
        return valid_transactions
    except requests.RequestException:
        st.error("Could not connect to the backend API. Make sure FastAPI is running.")
        return []
    except Exception:
        st.error("An unexpected error occurred while loading data.")
        return []


def save_transactions(transactions: List[Dict[str, Any]], filepath: str = DATA_FILE) -> bool:
    try:
        payload: List[Dict[str, Any]] = []
        for t in transactions:
            date_val = t.get("date")
            if isinstance(date_val, date):
                date_str = date_val.isoformat()
            else:
                date_str = str(date_val)
            try:
                amount = float(t.get("amount", 0.0))
            except (ValueError, TypeError):
                amount = 0.0
            payload.append(
                {
                    "id": t.get("id") or str(uuid.uuid4()),
                    "type": t.get("type"),
                    "amount": amount,
                    "category": t.get("category"),
                    "date": date_str,
                    "note": t.get("note", ""),
                }
            )

        response = requests.put(f"{API_URL}/transactions/bulk", json=payload, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        st.error("Failed to save transactions to the backend API.")
        return False
    except Exception:
        st.error("An unexpected error occurred while saving data.")
        return False


st.set_page_config(page_title="Budget Tracker", layout="wide", initial_sidebar_state="expanded")
inject_global_styles()
inject_sidebar_primary_button_styles()

if "transactions" not in st.session_state:
    st.session_state.transactions = load_transactions()
    st.session_state.transactions_snapshot = st.session_state.transactions.copy()

# After adding a row I set a flag and rerun; this block runs first so I can reset the form.
# (Can't assign widget keys after those widgets already drew — Streamlit throws an error.)
if st.session_state.pop("_reset_add_form", False):
    # clear everything back to defaults before the sidebar builds
    st.session_state["add_type"] = "Expense"
    st.session_state["add_category"] = None
    st.session_state["add_amount"] = 0.01
    st.session_state["add_date"] = datetime.now().date()
    st.session_state["add_note"] = ""

render_app_header()

with st.sidebar:
    st.header("Add New Transaction")
    transaction_type = st.selectbox("Type", ["Expense", "Income"], key="add_type")

    category_options = INCOME_CATEGORIES if transaction_type == "Income" else EXPENSE_CATEGORIES
    category = st.selectbox(
        "Category",
        options=category_options,
        index=None,
        placeholder="Select category...",
        key="add_category"
    )
    st.caption("Select a category from the list.")
    amount_raw = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f", key="add_amount")
    amount = round(amount_raw, 2)

    transaction_date = st.date_input("Date", key="add_date")
    
    note = st.text_area("Note (Optional)", key="add_note")

    form_errors: List[str] = []
    if not category:
        form_errors.append("Select a category.")
    if amount <= 0:
        form_errors.append("Enter an amount greater than 0.")

    if form_errors:
        st.caption("Please fix: " + " ".join(form_errors))

    is_adding = bool(st.session_state.get("_add_in_progress", False))
    add_label = "Adding..." if is_adding else "Add Transaction"
    if st.button(add_label, use_container_width=True, type="primary", disabled=is_adding or bool(form_errors)):
        if category and amount > 0:
            new_transaction = {
                "id": str(uuid.uuid4()),
                "type": transaction_type.lower(),
                "amount": amount,
                "category": category.strip(),
                "date": transaction_date.isoformat(),
                "note": note.strip(),
            }
            next_transactions = st.session_state.transactions.copy()
            next_transactions.insert(0, new_transaction)

            st.session_state["_add_in_progress"] = True
            with st.spinner("Adding transaction..."):
                ok = save_transactions(next_transactions)
            st.session_state["_add_in_progress"] = False

            if ok:
                st.session_state.transactions = next_transactions
                st.session_state.transactions_snapshot = next_transactions.copy()
                st.toast(f"{transaction_type} of {CURRENCY}{amount:.2f} added!", icon="✅")
                st.session_state["_reset_add_form"] = True
                st.rerun()
        elif not category:
            st.warning("Please select or enter a category.")
        elif amount <= 0:
            st.warning("Amount must be greater than zero.")

tab_summary, tab_history, tab_charts = st.tabs(["Summary", "History", "Charts"])

with tab_summary:
    st.subheader("Overview")

    if not st.session_state.transactions:
        st.info("Start by adding an income or expense on the left.")

    df_summary = pd.DataFrame(st.session_state.transactions)
    if not df_summary.empty:
        df_summary["amount"] = pd.to_numeric(df_summary["amount"])

        total_income = df_summary.loc[df_summary["type"] == "income", "amount"].sum()
        total_expense = df_summary.loc[df_summary["type"] == "expense", "amount"].sum()
        balance = total_income - total_expense

        today = datetime.now().date()
        first_this_month = today.replace(day=1)
        first_prev_month = (first_this_month - timedelta(days=1)).replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)

        df_summary["date_obj"] = pd.to_datetime(df_summary["date"]).dt.date

        this_month = df_summary[df_summary["date_obj"].between(first_this_month, today)]
        prev_month = df_summary[df_summary["date_obj"].between(first_prev_month, last_prev_month)]

        inc_this = this_month.loc[this_month["type"] == "income", "amount"].sum()
        exp_this = this_month.loc[this_month["type"] == "expense", "amount"].sum()
        net_this = inc_this - exp_this

        inc_prev = prev_month.loc[prev_month["type"] == "income", "amount"].sum()
        exp_prev = prev_month.loc[prev_month["type"] == "expense", "amount"].sum()
        net_prev = inc_prev - exp_prev

        net_delta = net_this - net_prev
    else:
        total_income = total_expense = balance = 0.0
        inc_this = exp_this = net_this = inc_prev = exp_prev = net_prev = net_delta = 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"{CURRENCY}{total_income:,.2f}")
    col2.metric("Total Expenses", f"{CURRENCY}{total_expense:,.2f}")
    col3.metric("Current Balance", f"{CURRENCY}{balance:,.2f}")

    st.markdown("###### This month vs last month")
    c1, c2, c3 = st.columns(3)
    c1.metric("This Month Income", f"{CURRENCY}{inc_this:,.2f}", delta=f"{CURRENCY}{inc_this - inc_prev:,.2f}")
    c2.metric("This Month Expenses", f"{CURRENCY}{exp_this:,.2f}", delta=f"{CURRENCY}{exp_this - exp_prev:,.2f}")
    c3.metric("This Month Net", f"{CURRENCY}{net_this:,.2f}", delta=f"{CURRENCY}{net_delta:,.2f}")

def _date_group_label(d: date) -> str:
    today = date.today()
    if d == today:
        return "Today"
    if d == today - timedelta(days=1):
        return "Yesterday"
    return d.strftime("%b %d, %Y")


with tab_history:
    st.subheader("Activity")

    if not st.session_state.transactions:
        st.info("No transactions yet. Add your first income or expense from the left panel.")
    else:
        tx_list = st.session_state.transactions
        today = date.today()
        all_cats = sorted(list(set(ALL_CATEGORIES + [t.get("category", "Other") for t in tx_list])))

        col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2.0])
        with col_f1:
            history_type_filter = st.selectbox(
                "Filter type",
                ["All", "Income", "Expense"],
                key="history_type_filter",
            )
        with col_f2:
            history_category_filter = st.selectbox(
                "Filter category",
                ["All categories"] + all_cats,
                key="history_category_filter",
            )
        with col_f3:
            history_search = st.text_input(
                "Search",
                placeholder="Search category or note...",
                key="history_search",
            ).strip().lower()

        filtered_tx_list: List[Dict[str, Any]] = []
        for t in tx_list:
            t_type = str(t.get("type", "")).lower()
            t_cat = str(t.get("category", "Other")).strip()
            t_note = str(t.get("note", "")).strip()
            if history_type_filter != "All" and t_type != history_type_filter.lower():
                continue
            if history_category_filter != "All categories" and t_cat != history_category_filter:
                continue
            if history_search and history_search not in t_cat.lower() and history_search not in t_note.lower():
                continue
            filtered_tx_list.append(t)

        if not filtered_tx_list:
            st.info("No transactions match your current filters.")
        else:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for t in filtered_tx_list:
                d = t.get("date", "")
                if isinstance(d, date):
                    d_key = d.isoformat()
                else:
                    d_key = str(d)[:10]
                try:
                    datetime.strptime(d_key, "%Y-%m-%d")
                except Exception:
                    d_key = today.isoformat()
                if d_key not in groups:
                    groups[d_key] = []
                groups[d_key].append(t)

            sorted_dates = sorted(groups.keys(), reverse=True)

            for d_key in sorted_dates:
                try:
                    d_obj = datetime.strptime(d_key, "%Y-%m-%d").date()
                except Exception:
                    d_obj = today
                display_label = _date_group_label(d_obj)
                st.markdown(f'<p class="statement-date-label">{display_label}</p>', unsafe_allow_html=True)
                for tx in groups[d_key]:
                    tx_id = tx.get("id", "")
                    amount = float(tx.get("amount", 0))
                    is_income = (tx.get("type") or "").lower() == "income"
                    category = (tx.get("category") or "Other").strip()
                    note = (tx.get("note") or "").strip()
                    amount_class = "statement-card-amount-income" if is_income else "statement-card-amount-expense"
                    sign = "+" if is_income else "−"
                    badge_class = "statement-badge-income" if is_income else "statement-badge-expense"
                    type_label = "Income" if is_income else "Expense"
                    meta = note if note else type_label
                    st.markdown(
                        f"""
                        <div class="statement-card">
                            <div class="statement-card-left">
                                <div class="statement-card-header">
                                    <span class="statement-card-category">{category}</span>
                                    <span class="statement-badge {badge_class}">{type_label}</span>
                                </div>
                                <div class="statement-card-meta">{meta}</div>
                            </div>
                            <div class="{amount_class}">{sign}{CURRENCY}{amount:,.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Edit or delete", expanded=False):
                        idx_in_list = next((i for i, x in enumerate(st.session_state.transactions) if x.get("id") == tx_id), None)
                        if idx_in_list is None:
                            continue
                        edit_type = st.selectbox(
                            "Type",
                            ["income", "expense"],
                            index=0 if is_income else 1,
                            key=f"hist_type_{tx_id}",
                        )
                        edit_category = st.selectbox(
                            "Category",
                            all_cats,
                            index=all_cats.index(category) if category in all_cats else 0,
                            key=f"hist_cat_{tx_id}",
                        )
                        edit_amount = st.number_input(
                            "Amount",
                            min_value=0.01,
                            step=0.01,
                            value=float(amount),
                            format="%.2f",
                            key=f"hist_amt_{tx_id}",
                        )
                        edit_date = st.date_input(
                            "Date",
                            value=datetime.strptime(tx.get("date", today.isoformat())[:10], "%Y-%m-%d").date()
                            if isinstance(tx.get("date"), str)
                            else today,
                            key=f"hist_date_{tx_id}",
                        )
                        edit_note = st.text_input("Note", value=note, key=f"hist_note_{tx_id}")
                        col_s, col_d = st.columns(2)
                        with col_s:
                            save_clicked = st.button("Save changes", key=f"save_{tx_id}")
                        with col_d:
                            pending_delete_id = st.session_state.get("_pending_delete")
                            if pending_delete_id == tx_id:
                                confirm_delete_clicked = st.button(
                                    "Confirm delete", key=f"confirm_del_{tx_id}"
                                )
                                cancel_delete_clicked = st.button(
                                    "Cancel", key=f"cancel_del_{tx_id}"
                                )
                            else:
                                confirm_delete_clicked = False
                                cancel_delete_clicked = False
                                del_clicked = st.button("Delete", key=f"del_{tx_id}")
                        if save_clicked:
                            new_tx = {
                                "id": tx_id,
                                "type": edit_type,
                                "amount": round(edit_amount, 2),
                                "category": edit_category,
                                "date": edit_date.isoformat(),
                                "note": edit_note.strip(),
                            }
                            next_tx_list = st.session_state.transactions.copy()
                            next_tx_list[idx_in_list] = new_tx
                            with st.spinner("Saving changes..."):
                                ok = save_transactions(next_tx_list)
                            if ok:
                                st.session_state.transactions = next_tx_list
                                st.session_state.transactions_snapshot = next_tx_list.copy()
                                st.toast("Transaction updated.")
                                st.rerun()

                        # delete: first click asks, second click actually deletes
                        if st.session_state.get("_pending_delete") == tx_id:
                            if cancel_delete_clicked:
                                st.session_state["_pending_delete"] = None
                                st.rerun()
                            elif confirm_delete_clicked:
                                next_tx_list = st.session_state.transactions.copy()
                                next_tx_list.pop(idx_in_list)
                                with st.spinner("Deleting transaction..."):
                                    ok = save_transactions(next_tx_list)
                                if ok:
                                    st.session_state.transactions = next_tx_list
                                    st.session_state.transactions_snapshot = next_tx_list.copy()
                                    st.session_state["_pending_delete"] = None
                                    st.toast("Transaction removed.")
                                    st.rerun()
                        else:
                            if del_clicked:
                                st.session_state["_pending_delete"] = tx_id
                                st.rerun()

        st.markdown("---")
        df_export = pd.DataFrame(st.session_state.transactions)
        df_export["date"] = pd.to_datetime(df_export["date"]).dt.date
        df_export["amount"] = pd.to_numeric(df_export["amount"])
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"budget_transactions_{date.today()}.csv",
            mime="text/csv",
        )

with tab_charts:
    st.subheader("Visualizations")

    if not st.session_state.transactions:
        st.info("No transaction data to display charts.")
    else:
        df_charts = pd.DataFrame(st.session_state.transactions)
        df_charts["date"] = pd.to_datetime(df_charts["date"])
        df_charts["amount"] = pd.to_numeric(df_charts["amount"])

        min_chart_date = df_charts["date"].min().date() if not df_charts.empty else date.today() - timedelta(days=30)
        max_chart_date = df_charts["date"].max().date() if not df_charts.empty else date.today()

        st.markdown("##### Date range")
        preset = st.radio(
            "Select a quick range or choose Custom for exact dates.",
            options=["Last 7 days", "Last 30 days", "This month", "This year", "All time", "Custom"],
            index=1,
            horizontal=True,
        )

        today = date.today()
        if preset == "Last 7 days":
            chart_start_date = max(min_chart_date, today - timedelta(days=7))
            chart_end_date = today
        elif preset == "Last 30 days":
            chart_start_date = max(min_chart_date, today - timedelta(days=30))
            chart_end_date = today
        elif preset == "This month":
            first_this_month = today.replace(day=1)
            chart_start_date = max(min_chart_date, first_this_month)
            chart_end_date = today
        elif preset == "This year":
            first_this_year = date(today.year, 1, 1)
            chart_start_date = max(min_chart_date, first_this_year)
            chart_end_date = today
        elif preset == "All time":
            chart_start_date = min_chart_date
            chart_end_date = max_chart_date
        else:
            col_c_start, col_c_end = st.columns(2)
            with col_c_start:
                chart_start_date = st.date_input(
                    "Start date",
                    value=min_chart_date,
                    min_value=min_chart_date,
                    max_value=max_chart_date,
                    key="chart_custom_start",
                )
            with col_c_end:
                chart_end_date = st.date_input(
                    "End date",
                    value=max_chart_date,
                    min_value=min_chart_date,
                    max_value=max_chart_date,
                    key="chart_custom_end",
                )

        mask = (df_charts["date"].dt.date >= chart_start_date) & (df_charts["date"].dt.date <= chart_end_date)
        df_filtered_charts = df_charts.loc[mask]

        if df_filtered_charts.empty:
            st.warning("No transactions found in the selected date range for charts.")
        else:
            st.markdown("#### Expenses by Category")
            expense_df = df_filtered_charts[df_filtered_charts['type'] == 'expense']
            if not expense_df.empty:
                expense_summary = expense_df.groupby('category')['amount'].sum().reset_index()
                expense_summary = expense_summary.sort_values(by='amount', ascending=False)

                fig_pie = px.pie(
                    expense_summary,
                    values="amount",
                    names="category",
                    hole=0.4,
                    title=f"Expense Distribution ({CURRENCY})",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    legend_title_text="Categories",
                    margin=dict(t=60, b=0, l=0, r=0),
                    template="streamlit",
                )
                st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")
            else:
                st.info("No expenses found in this period.")

            st.markdown("#### Monthly Income vs. Expense Trend")
            df_filtered_charts["Month"] = df_filtered_charts["date"].dt.to_period("M").astype(str)
            trend_summary = (
                df_filtered_charts.groupby(["Month", "type"])["amount"]
                .sum()
                .unstack(fill_value=0)
                .reset_index()
            )

            if 'income' not in trend_summary.columns:
                trend_summary['income'] = 0.0
            if 'expense' not in trend_summary.columns:
                trend_summary['expense'] = 0.0

            trend_summary = trend_summary.sort_values(by='Month')

            fig_bar = px.bar(
                trend_summary,
                x="Month",
                y=["income", "expense"],
                title="Income vs. Expense Over Time",
                barmode="group",
                labels={"value": f"Amount ({CURRENCY})", "variable": "Transaction Type"},
                color_discrete_map={"income": "#10b981", "expense": "#ef4444"},
            )
            fig_bar.update_layout(
                yaxis_title=f"Amount ({CURRENCY})",
                xaxis_title="Month",
                legend_title_text="Type",
                margin=dict(t=60, b=40, l=40, r=20),
                template="streamlit",
            )
            st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")
            
            st.markdown("#### Cumulative Balance Over Time")
            df_balance = df_filtered_charts.copy()
            df_balance = df_balance.sort_values(by="date")
            df_balance["change"] = df_balance.apply(
                lambda row: row["amount"] if row["type"] == "income" else -row["amount"], axis=1
            )
            df_balance["cumulative_balance"] = df_balance["change"].cumsum()
            
            fig_line = px.line(
                df_balance,
                x="date",
                y="cumulative_balance",
                title="Cumulative Balance Trend",
                labels={"date": "Date", "cumulative_balance": f"Balance ({CURRENCY})"},
                markers=True
            )
            fig_line.update_layout(
                yaxis_title=f"Balance ({CURRENCY})",
                xaxis_title="Date",
                margin=dict(t=60, b=40, l=40, r=20),
                template="streamlit",
            )
            st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")