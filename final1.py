from fastapi import FastAPI
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

# ------------------------------
# Database path
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / '../../subscription_management.db'

# ------------------------------
# Utility: Fetch data from DB
# ------------------------------
def fetch_user_subscription_data():
    """Fetch users and their subscription info as DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT u.id AS user_id, u.full_name AS name, u.role AS status,
           s.id AS subscription_id, s.start_date, s.end_date, s.data_used,
           sp.name AS plan_name, sp.data_quota, sp.price AS plan_price, sp.duration_days
    FROM users u
    LEFT JOIN subscriptions s ON u.id = s.user_id
    LEFT JOIN subscription_plans sp ON s.plan_id = sp.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert dates
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['UsageRatio'] = df['data_used'] / df['data_quota']
    df['UsageRatio'] = df['UsageRatio'].fillna(0)

    return df

# ------------------------------
# ML: Cluster-based recommendation
# ------------------------------
def cluster_recommendation(df):
    """Cluster users by usage and plan price for recommendations"""
    features = df[['data_used', 'plan_price']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_plan = df.groupby('Cluster')['data_quota'].agg(lambda x: x.value_counts().index[0])

    def cluster_based(row):
        X_new = pd.DataFrame([[row['data_used'], row['plan_price']]], columns=['data_used', 'plan_price'])
        cluster = kmeans.predict(scaler.transform(X_new))[0]
        return f"Users like you prefer plan with {cluster_plan[cluster]} GB quota"

    df['ClusterReco'] = df.apply(cluster_based, axis=1)
    return df

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="Subscription Analytics API")

# ------------------------------
# USER: Individual Recommendations
# ------------------------------
@app.get("/recommendations/{user_id}")
async def user_recommendation(user_id: str):
    df = fetch_user_subscription_data()

    user = df[df['user_id'] == user_id]
    if user.empty:
        return {"error": "User not found"}

    # Rule-based recommendation
    def rule_based(row):
        if row['status'].lower() == 'enduser':
            if row['UsageRatio'] > 0.8:
                return "Upgrade Plan"
            elif row['UsageRatio'] < 0.3:
                return "Downgrade Plan"
            else:
                return "Keep Current Plan"
        return "Admin"

    user['Recommendation'] = user.apply(rule_based, axis=1)

    # Cluster-based recommendation
    user = cluster_recommendation(user)

    # Renewal notification
    today = pd.Timestamp(datetime.now())
    user['RenewalDue'] = user['end_date'].apply(lambda x: (x - today).days if pd.notnull(x) else None)

    return user[['user_id', 'name', 'status', 'plan_name', 'data_quota', 'data_used',
                 'UsageRatio', 'Recommendation', 'ClusterReco', 'RenewalDue']].to_dict(orient='records')[0]

# ------------------------------
# ADMIN: All Users and Analytics
# ------------------------------
@app.get("/admin/recommendations")
async def admin_recommendations():
    df = fetch_user_subscription_data()

    # Rule-based recommendation
    def rule_based(row):
        if row['status'].lower() == 'enduser':
            if row['UsageRatio'] > 0.8:
                return "Upgrade Plan"
            elif row['UsageRatio'] < 0.3:
                return "Downgrade Plan"
            else:
                return "Keep Current Plan"
        return "Admin"

    df['Recommendation'] = df.apply(rule_based, axis=1)
    df = cluster_recommendation(df)

    # Renewal alerts
    today = pd.Timestamp(datetime.now())
    df['RenewalDue'] = df['end_date'].apply(lambda x: (x - today).days if pd.notnull(x) else None)
    df['RenewalAlert'] = df['RenewalDue'].apply(lambda x: x <= 7 if x is not None else False)

    # Analytics summary
    plan_summary = df.groupby('plan_name')['user_id'].count().to_dict()
    avg_usage = round(df['UsageRatio'].mean(), 2)
    renewals_due = df[df['RenewalAlert'] == True][['user_id', 'name', 'plan_name', 'RenewalDue']].to_dict(orient='records')

    return {
        "users": df[['user_id', 'name', 'status', 'plan_name', 'data_quota', 'data_used',
                     'UsageRatio', 'Recommendation', 'ClusterReco', 'RenewalDue']].to_dict(orient='records'),
        "plan_summary": plan_summary,
        "average_usage_ratio": avg_usage,
        "renewals_due_soon": renewals_due
    }

# ------------------------------
# ROOT
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Subscription Analytics API is running 🚀"}
