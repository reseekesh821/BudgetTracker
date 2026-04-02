💵 **Personal Budget Tracker**

This is a personal finance tracking app built with **Streamlit** and a **FastAPI** backend. It lets you add income and expenses, review summaries, filter your history, and visualize spending trends. Data is stored in a local **SQLite** database (`data/budget.db`).

## Features

- **Add, edit, and delete** income or expense transactions (with confirmation before delete).
- **Summary dashboard**: total income, expenses, and net balance, including **this month vs last month**.
- **History**: filter by type and category, search by category or note, and export your list.
- **Visualizations**: pie chart (spending by category), bar chart (monthly income vs expense), and line chart (cumulative balance)—styled to match the app theme.
- **Export**: download transaction history as **CSV**.
- **Optional admin app**: inspect and clean records in the database (`admin_app.py`).

## How to run

1. Make sure **Python 3.11+** is installed.

2. Install dependencies (from the project folder):

   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Start the FastAPI backend** (creates/uses the SQLite DB and API):

   ```bash
   python -m uvicorn api:app --reload
   ```

4. In a **second** terminal, **start the main Streamlit app**:

   ```bash
   python -m streamlit run app.py
   ```

5. **Optional** — run the admin UI in another terminal:

   ```bash
   python -m streamlit run admin_app.py
   ```

The main app is the budgeting dashboard; the admin app is for database-focused management.

## File structure

- `app.py` — Main Streamlit dashboard (add form, summary, history, charts, CSV export).
- `api.py` — FastAPI backend (SQLite + REST API for transactions).
- `admin_app.py` — Optional admin interface for the database.
- `data/budget.db` — SQLite database (created when the API runs).
- `requirements.txt` — Python dependencies.
