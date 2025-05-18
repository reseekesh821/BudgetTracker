💵 Personal Budget Tracker

This is a personal finance tracking app built using Streamlit. It allows users to manage income and expenses, view financial summaries, and visualize spending trends. All data is stored in a local JSON file.

# Features

1. Add, edit, and delete income or expense transactions

2. View summaries of total income, expenses, and net balance

3. Visualizations: pie chart (spending by category), bar chart (monthly trends), and line chart (cumulative balance)

4. Save and load transactions from a JSON file

5. Export transaction history as CSV

# How to Run

1. Make sure Python is installed.

2. Install dependencies:
   pip install streamlit plotly pandas

3. Run the app:
   python -m streamlit run app.py

# File Structure

app.py – Main application file

data/transactions.json – Stores transaction data automatically.