import pandas as pd
import random
from datetime import datetime, timedelta

# 1. THE BASELINE (Control)
def generate_baseline_profile(start_date_str="2026-01-01", days=90):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    transactions = []
    date_range = [start_date + timedelta(days=x) for x in range(days)]
    
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        if current_date.toordinal() % 14 == 0:
            transactions.append({"date": date_str, "type": "income", "amount": 2500.00, "category": "Salary", "note": "Bi-weekly paycheck"})
        if current_date.day == 1:
            transactions.append({"date": date_str, "type": "expense", "amount": 1200.00, "category": "Rent", "note": "Rent payment"})
        if random.random() < 0.15:
            transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(50, 150), 2), "category": "Groceries", "note": "Supermarket"})
        if random.random() < 0.30:
            categories = ["Dining Out", "Transport", "Entertainment"]
            transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(5, 45), 2), "category": random.choice(categories), "note": "Misc expense"})
    return pd.DataFrame(transactions)

# 2. THE IMPULSIVE SPENDER
def generate_impulsive_profile(start_date_str="2026-01-01", days=90):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    transactions = []
    date_range = [start_date + timedelta(days=x) for x in range(days)]
    
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        if current_date.toordinal() % 14 == 0:
            transactions.append({"date": date_str, "type": "income", "amount": 2500.00, "category": "Salary", "note": "Bi-weekly paycheck"})
        if current_date.day == 1:
            transactions.append({"date": date_str, "type": "expense", "amount": 1200.00, "category": "Rent", "note": "Rent payment"})
        
        is_weekend = current_date.weekday() in [4, 5, 6]
        if is_weekend:
            if random.random() < 0.85: 
                for _ in range(random.randint(1, 4)):
                    impulse_categories = ["Dining Out", "Entertainment", "Clothing"]
                    transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(30.00, 150.00), 2), "category": random.choice(impulse_categories), "note": "Weekend impulse buy"})
        else:
            if random.random() < 0.20:
                transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(10.00, 25.00), 2), "category": "Dining Out", "note": "Weekday lunch"})
    return pd.DataFrame(transactions)

# 3. SUBSCRIPTION CREEP
def generate_creep_profile(start_date_str="2026-01-01", days=90):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    transactions = []
    date_range = [start_date + timedelta(days=x) for x in range(days)]
    
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        if current_date.toordinal() % 14 == 0:
            transactions.append({"date": date_str, "type": "income", "amount": 2500.00, "category": "Salary", "note": "Bi-weekly paycheck"})
        if current_date.day == 1:
            transactions.append({"date": date_str, "type": "expense", "amount": 1200.00, "category": "Rent", "note": "Rent payment"})
            
        if current_date.day == 5:
            amt = 9.99 if current_date.month == 1 else (12.99 if current_date.month == 2 else 15.99)
            transactions.append({"date": date_str, "type": "expense", "amount": amt, "category": "Entertainment", "note": "StreamFlix Sub"})
            
        if current_date.day == 15:
            amt = 4.99 if current_date.month <= 2 else 19.99
            transactions.append({"date": date_str, "type": "expense", "amount": amt, "category": "Utilities", "note": "Cloud Storage Pro"})

        if current_date.day == 20:
            amt = 45.00 if current_date.month == 1 else (48.50 if current_date.month == 2 else 52.00)
            transactions.append({"date": date_str, "type": "expense", "amount": amt, "category": "Healthcare", "note": "Gym Fees"})
            
        if random.random() < 0.15: 
            transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(15.00, 35.00), 2), "category": "Groceries", "note": "Quick mart run"})
    return pd.DataFrame(transactions)

# 4. GIG WORKER (Variable Income)
def generate_variable_profile(start_date_str="2026-01-01", days=90):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    transactions = []
    date_range = [start_date + timedelta(days=x) for x in range(days)]
    
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        
        if random.random() < 0.10:
            transactions.append({"date": date_str, "type": "income", "amount": round(random.uniform(150.00, 900.00), 2), "category": "Freelance", "note": "Gig payout"})
            
        if current_date.day == 1:
            transactions.append({"date": date_str, "type": "expense", "amount": 1200.00, "category": "Rent", "note": "Rent payment"})
            
        if random.random() < 0.20:
            transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(20.00, 80.00), 2), "category": "Groceries", "note": "Food run"})
        if random.random() < 0.15:
            transactions.append({"date": date_str, "type": "expense", "amount": round(random.uniform(15.00, 40.00), 2), "category": "Transport", "note": "Gas/Transit for gigs"})
    return pd.DataFrame(transactions)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    profiles = {
        "baseline_profile.csv": generate_baseline_profile,
        "impulsive_profile.csv": generate_impulsive_profile,
        "creep_profile.csv": generate_creep_profile,
        "variable_profile.csv": generate_variable_profile
    }
    
    for filename, func in profiles.items():
        df = func()
        df = df.sort_values(by="date")
        df.to_csv(filename, index=False)
        print(f"Successfully generated {filename} with {len(df)} records.")