import os
import json
import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
import plotly.express as px
import uuid
import logging
from typing import List, Dict, Any, Optional

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "transactions.json")
INCOME_CATEGORIES = ["Salary", "Freelance", "Investment", "Gift", "Bonus", "Other"]
EXPENSE_CATEGORIES = ["Groceries", "Rent", "Utilities", "Dining Out", "Transport", "Entertainment", "Clothing", "Healthcare", "Other"]
ALL_CATEGORIES = sorted(list(set(INCOME_CATEGORIES + EXPENSE_CATEGORIES)))
CURRENCY = "$" # Configurable currency symbol

os.makedirs(DATA_DIR, exist_ok=True)

# Data Handling Functions
def load_transactions(filepath: str = DATA_FILE) -> List[Dict[str, Any]]:
    """Load and validate transactions from a JSON file."""
    transactions = []
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        logging.info(f"Data file '{filepath}' not found or empty. Starting fresh.")
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.warning(f"Data in '{filepath}' is not a list. Returning empty list.")
            return []

        needs_saving = False
        valid_transactions = []
        for idx, t in enumerate(data):
            if not isinstance(t, dict):
                logging.warning(f"Item at index {idx} is not a dictionary. Skipping.")
                continue

            # Ensure essential keys and types
            t_id = t.get('id')
            if not t_id or not isinstance(t_id, str):
                t['id'] = str(uuid.uuid4())
                logging.info(f"Assigned new UUID to transaction: {t}")
                needs_saving = True
                
            t_type = t.get('type')
            if t_type not in ['income', 'expense']:
                logging.warning(f"Invalid or missing type for transaction {t.get('id', 'N/A')}. Skipping.")
                continue
                
            try:
                t['amount'] = abs(float(t.get('amount', 0.0))) # Ensure positive float
                if t['amount'] == 0.0:
                     logging.warning(f"Transaction {t.get('id', 'N/A')} has zero amount. Skipping.")
                     continue
            except (ValueError, TypeError):
                logging.warning(f"Invalid amount for transaction {t.get('id', 'N/A')}. Skipping.")
                continue

            t['category'] = str(t.get('category', 'Other')).strip()
            if not t['category']:
                 t['category'] = 'Other'
                 logging.warning(f"Empty category for transaction {t.get('id', 'N/A')}. Set to 'Other'.")

            try:
                # Ensure date is stored as ISO format string
                date_val = t.get('date')
                if isinstance(date_val, date):
                     t['date'] = date_val.isoformat()
                elif isinstance(date_val, str):
                     # Validate ISO format
                     datetime.strptime(date_val, '%Y-%m-%d')
                     t['date'] = date_val # Already string
                else:
                     logging.warning(f"Invalid or missing date for transaction {t.get('id', 'N/A')}. Setting to today.")
                     t['date'] = datetime.now().date().isoformat()
                     needs_saving = True # Corrected the date format
            except ValueError:
                 logging.warning(f"Invalid date format for transaction {t.get('id', 'N/A')}. Setting to today.")
                 t['date'] = datetime.now().date().isoformat()
                 needs_saving = True # Corrected the date format

            t['note'] = str(t.get('note', '')).strip()

            valid_transactions.append(t)

        if needs_saving:
            logging.info("Updating data file due to corrections during load.")
            save_transactions(valid_transactions, filepath)

        # Sort by date descending after loading
        valid_transactions.sort(key=lambda x: x.get('date', '1970-01-01'), reverse=True)
        logging.info(f"Loaded {len(valid_transactions)} transactions successfully.")
        return valid_transactions

    except FileNotFoundError:
        logging.info(f"Data file '{filepath}' not found. Starting fresh.")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{filepath}'. File might be corrupted.")
        st.error(f"Error reading data file '{filepath}'. Please check its format.")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during loading: {e}", exc_info=True)
        st.error("An unexpected error occurred while loading data.")
        return []

def save_transactions(transactions: List[Dict[str, Any]], filepath: str = DATA_FILE):
    """Save transactions to a JSON file with proper formatting."""
    try:
        # Ensure dates are strings before saving
        for t in transactions:
            if isinstance(t.get('date'), date):
                t['date'] = t['date'].isoformat()
            # Ensure amount is float
            try:
                t['amount'] = float(t.get('amount', 0.0))
            except (ValueError, TypeError):
                t['amount'] = 0.0 # Default to 0 if conversion fails
                logging.warning(f"Could not convert amount to float for transaction {t.get('id', 'N/A')} during save. Setting to 0.0.")

        with open(filepath, "w") as f:
            json.dump(transactions, f, indent=4, default=str) # Use default=str as fallback
        logging.info(f"Saved {len(transactions)} transactions to '{filepath}'.")
    except Exception as e:
        logging.error(f"Error saving transactions to '{filepath}': {e}", exc_info=True)
        st.error(f"Failed to save transactions: {e}")

# Initialize Session State
if 'transactions' not in st.session_state:
    st.session_state.transactions = load_transactions()
    # Store the initial loaded state for comparison after edits
    st.session_state.transactions_snapshot = st.session_state.transactions.copy()

# Streamlit App Layout
st.set_page_config(page_title="Budget Tracker", layout="wide", initial_sidebar_state="expanded")
st.title("💵Personal Budget Tracker")

# Sidebar for Adding Transactions
## Sidebar form to input and save a new income or expense
with st.sidebar:
    st.header("Add New Transaction")
    transaction_type = st.selectbox("Type", ["Expense", "Income"], key="add_type")

    # Combine category selection with option for custom input
    category_options = INCOME_CATEGORIES if transaction_type == "Income" else EXPENSE_CATEGORIES
    category = st.selectbox(
        "Category",
        options=category_options,
        index=None,  # Default to no selection
        placeholder="Select category...", # Add placeholder text
        key="add_category"
    )
    st.caption("Select a category from the list.")
    amount_raw = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f", key="add_amount")
    amount = round(amount_raw, 2) # Ensure 2 decimal places

    transaction_date = st.date_input("Date", value=datetime.now().date(), key="add_date")
    
    note = st.text_area("Note (Optional)", key="add_note")

    if st.button("Add Transaction", use_container_width=True, type="primary"):
        if category and amount > 0:
            new_transaction = {
                "id": str(uuid.uuid4()),
                "type": transaction_type.lower(),
                "amount": amount,
                "category": category.strip(),
                "date": transaction_date.isoformat(), # Save as ISO string
                "note": note.strip(),
            }
            # Insert at the beginning for most recent first view
            st.session_state.transactions.insert(0, new_transaction)
            save_transactions(st.session_state.transactions)
            st.toast(f"{transaction_type} of {CURRENCY}{amount:.2f} added!", icon="✅")
            # Clear input fields by resetting keys or using form submission (rerun is simpler here)
            st.rerun()
        elif not category:
            st.warning("Please select or enter a category.")
        elif amount <= 0:
            st.warning("Amount must be greater than zero.")

# Main Content Tabs
tab_summary, tab_history, tab_charts = st.tabs(["📈 Summary", " 📝History", "📊 Charts"])

with tab_summary:
    st.subheader("Overall Financial Summary")
    
    df_summary = pd.DataFrame(st.session_state.transactions)
    if not df_summary.empty:
        df_summary['amount'] = pd.to_numeric(df_summary['amount'])
        total_income = df_summary.loc[df_summary['type'] == 'income', 'amount'].sum()
        total_expense = df_summary.loc[df_summary['type'] == 'expense', 'amount'].sum()
        balance = total_income - total_expense
    else:
        total_income = 0.0
        total_expense = 0.0
        balance = 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"{CURRENCY}{total_income:,.2f}", delta_color="normal")
    col2.metric("Total Expenses", f"{CURRENCY}{total_expense:,.2f}", delta_color="inverse")
    col3.metric("Net Balance", f"{CURRENCY}{balance:,.2f}", delta=f"{CURRENCY}{balance - 0:,.2f}") # Simple delta vs 0

# tab_history
## View, edit, or delete past transactions using a table editor
with tab_history:
    st.subheader("Transaction History & Editor")

    if not st.session_state.transactions:
        st.info("No transactions recorded yet. Add some using the sidebar!")
    else:
        df_history = pd.DataFrame(st.session_state.transactions)
        df_history['date'] = pd.to_datetime(df_history['date']).dt.date # Convert to date objects for display
        df_history['amount'] = pd.to_numeric(df_history['amount'])
        
        # Data Editor
        st.markdown("**Edit or Delete Transactions:**")
        editor_key = "data_editor_history" 
        
        # Define columns for the editor
        all_known_categories = sorted(list(set(ALL_CATEGORIES + df_history['category'].unique().tolist())))
        column_config = {
            "id": None, # Hide internal ID
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
            "type": st.column_config.SelectboxColumn("Type", options=["income", "expense"], required=True),
            "category": st.column_config.SelectboxColumn("Category", options=all_known_categories, required=True),
            "amount": st.column_config.NumberColumn("Amount", format=f"{CURRENCY}%.2f", required=True, min_value=0.01),
            "note": st.column_config.TextColumn("Note"),
        }

        # Display the data editor with all transactions
        edited_df = st.data_editor(
            df_history,
            key=editor_key,
            num_rows="dynamic", # Allow adding/deleting rows
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            disabled=['id'] # Ensure ID is not editable
        )

        # Save Edited Data
        edited_transactions = edited_df.to_dict('records')
        
        # Process edits: Convert date back to string, ensure float amount, add UUIDs to new rows
        processed_edited_transactions = []
        ids_seen = set()
        save_needed = False
        
        for t in edited_transactions:
            if 'id' not in t or not t['id'] or not isinstance(t['id'], str) or len(t['id']) < 10:
                t['id'] = str(uuid.uuid4())
                save_needed = True
                
            if isinstance(t.get('date'), date):
                t['date'] = t['date'].isoformat()
                
            try:
                t['amount'] = round(abs(float(t.get('amount', 0.0))), 2)
            except (ValueError, TypeError):
                t['amount'] = 0.0

            if t['id'] not in ids_seen:
                processed_edited_transactions.append(t)
                ids_seen.add(t['id'])

        processed_edited_transactions.sort(key=lambda x: x.get('date', '1970-01-01'), reverse=True)
        
        # Check if changes were made
        current_snapshot = st.session_state.transactions_snapshot
        set_processed = set(tuple(sorted(d.items())) for d in processed_edited_transactions)
        set_snapshot = set(tuple(sorted(d.items())) for d in current_snapshot)

        if set_processed != set_snapshot or save_needed:
            st.session_state.transactions = processed_edited_transactions
            save_transactions(st.session_state.transactions)
            st.session_state.transactions_snapshot = st.session_state.transactions.copy()
            st.toast("Changes saved successfully!", icon="💾")
            st.rerun()

        # Data Export
        st.markdown("---")
        st.markdown("**Export Data:**")
        
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Transactions as CSV",
            data=csv_data,
            file_name=f"budget_transactions_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

# chart
## Show visual charts of spending and income trends over time
with tab_charts:
    st.subheader("Visualizations")

    if not st.session_state.transactions:
        st.info("No transaction data to display charts.")
    else:
        df_charts = pd.DataFrame(st.session_state.transactions)
        df_charts['date'] = pd.to_datetime(df_charts['date'])
        df_charts['amount'] = pd.to_numeric(df_charts['amount'])
        
        # Date Range Filter for Charts
        min_chart_date = df_charts['date'].min().date() if not df_charts.empty else date.today() - timedelta(days=30)
        max_chart_date = df_charts['date'].max().date() if not df_charts.empty else date.today()

        col_c_start, col_c_end = st.columns(2)
        with col_c_start:
            chart_start_date = st.date_input("Start Date", value=min_chart_date, min_value=min_chart_date, max_value=max_chart_date, key="chart_start")
        with col_c_end:
            chart_end_date = st.date_input("End Date", value=max_chart_date, min_value=min_chart_date, max_value=max_chart_date, key="chart_end")

        mask = (df_charts['date'].dt.date >= chart_start_date) & (df_charts['date'].dt.date <= chart_end_date)
        df_filtered_charts = df_charts.loc[mask]

        if df_filtered_charts.empty:
            st.warning("No transactions found in the selected date range for charts.")
        else:
            # Expense Pie Chart
            st.markdown("#### Expenses by Category")
            expense_df = df_filtered_charts[df_filtered_charts['type'] == 'expense']
            if not expense_df.empty:
                expense_summary = expense_df.groupby('category')['amount'].sum().reset_index()
                # Sort by amount descending for better pie visualization
                expense_summary = expense_summary.sort_values(by='amount', ascending=False)

                fig_pie = px.pie(expense_summary, 
                                 values='amount', 
                                 names='category', 
                                 hole=0.00, 
                                 title=f"Expense Distribution ({CURRENCY})",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(expense_summary)) # Slight pull effect
                fig_pie.update_layout(legend_title_text='Categories')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No expenses found in this period.")

            # Income vs Expense Bar Chart
            st.markdown("#### Monthly Income vs. Expense Trend")
            df_filtered_charts['Month'] = df_filtered_charts['date'].dt.to_period('M').astype(str)
            trend_summary = df_filtered_charts.groupby(['Month', 'type'])['amount'].sum().unstack(fill_value=0).reset_index()

            # Ensure both income and expense columns exist
            if 'income' not in trend_summary.columns:
                trend_summary['income'] = 0.0
            if 'expense' not in trend_summary.columns:
                trend_summary['expense'] = 0.0

            # Sort by month for chronological order
            trend_summary = trend_summary.sort_values(by='Month')

            fig_bar = px.bar(
                trend_summary,
                x='Month',
                y=['income', 'expense'],
                title="Income vs. Expense Over Time",
                barmode='group', # 'overlay' or 'relative' are alternatives
                labels={'value': f'Amount ({CURRENCY})', 'variable': 'Transaction Type'},
                color_discrete_map={'income': 'mediumseagreen', 'expense': 'lightcoral'} # Updated colors
            )
            fig_bar.update_layout(yaxis_title=f'Amount ({CURRENCY})', xaxis_title='Month', legend_title_text='Type')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Cumulative Balance Trend (Optional Line Chart) 
            st.markdown("#### Cumulative Balance Over Time")
            df_balance = df_filtered_charts.copy()
            df_balance = df_balance.sort_values(by='date')
            df_balance['change'] = df_balance.apply(lambda row: row['amount'] if row['type'] == 'income' else -row['amount'], axis=1)
            df_balance['cumulative_balance'] = df_balance['change'].cumsum()
            
            fig_line = px.line(
                df_balance,
                x='date',
                y='cumulative_balance',
                title="Cumulative Balance Trend",
                labels={'date': 'Date', 'cumulative_balance': f'Balance ({CURRENCY})'},
                markers=True
            )
            fig_line.update_layout(yaxis_title=f'Balance ({CURRENCY})', xaxis_title='Date')
            st.plotly_chart(fig_line, use_container_width=True)

