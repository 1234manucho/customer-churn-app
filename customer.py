import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sqlite3

# Database setup
DATABASE_NAME = "customer_churn.db"

def create_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            churn INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_sample_data():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    sample_data = [
        ("Alice", 25, 12, 70.5, 850.0, 0),
        ("Bob", 45, 24, 60.0, 1440.0, 1),
        ("Charlie", 35, 6, 80.0, 480.0, 0),
        ("Diana", 50, 36, 90.0, 3240.0, 1),
        ("Eve", 29, 18, 65.0, 1170.0, 0)
    ]
    cursor.executemany('''
        INSERT INTO customers (name, age, tenure, monthly_charges, total_charges, churn)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_data)
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DATABASE_NAME)
    query = "SELECT age, tenure, monthly_charges, total_charges, churn FROM customers"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_model(data):
    X = data.drop("churn", axis=1)
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

def main():
    create_database()
    insert_sample_data()
    data = load_data()
    print("Loaded Data:\n", data)
    model = train_model(data)
    print("Model training complete.")

if __name__ == "__main__":
    main()