from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------
# Helper: Churn Prediction
# ------------------------------
def add_churn_prediction(df: pd.DataFrame):
    """Adds churn prediction column to dataframe"""
    if df.empty:
        df["ChurnPrediction"] = None
        return df

    # Create synthetic churn label for training
    df['churn_label'] = (df['UsageRatio'] < 0.2).astype(int)

    # Select features
    X = df[['data_used', 'plan_price', 'data_quota']].fillna(0)
    y = df['churn_label']

    if y.nunique() == 1:  # if all labels are the same
        df['ChurnPrediction'] = y
        return df

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on full dataset
    df['ChurnPrediction'] = model.predict(X)

    return df.drop(columns=['churn_label'])
from fastapi import FastAPI
from supabase import create_client
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# Supabase Client
# ------------------------------
SUPABASE_URL = "https://yzjvfkazpqevuxxmtjsf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl6anZma2F6cHFldnV4eG10anNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc3NDk5OTQsImV4cCI6MjA3MzMyNTk5NH0.c2bc6MU8UVleG8yY7BrGw6FAKPSXCfPBK--VjASxt-A"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="Python Subscription Analytics API")

# ------------------------------
# Helper: Fetch subscriptions
# ------------------------------
def fetch_subscriptions():
    """Fetch subscriptions joined with users and plans"""
    response = supabase.table("subscriptions").select(
        "id, data_used, start_date, end_date, auto_renew, "
        "users(id, full_name, role), subscription_plans(id, name, data_quota, price)"
    ).execute()

    if not response.data:
        return None

    # Flatten
    data = []
    for sub in response.data:
        u = sub.get("users", {})
        p = sub.get("subscription_plans", {})
        data.append({
            "subscription_id": sub["id"],
            "user_id": u.get("id"),
            "name": u.get("full_name"),
            "role": u.get("role"),
            "plan_id": p.get("id"),
            "plan_name": p.get("name"),
            "data_quota": p.get("data_quota", 1),   # avoid div/0
            "plan_price": p.get("price", 0),
            "data_used": sub.get("data_used", 0),
            "start_date": sub.get("start_date"),
            "end_date": sub.get("end_date"),
            "auto_renew": sub.get("auto_renew")
        })
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["UsageRatio"] = df["data_used"] / df["data_quota"]
    df["UsageRatio"] = df["UsageRatio"].fillna(0)
    return df

# ------------------------------
# Rule-based recommendation
# ------------------------------
def rule_based_recommendation(row):
    if row["role"] and row["role"].lower() == "enduser":
        if row["UsageRatio"] > 0.8:
            return "Upgrade Plan"
        elif row["UsageRatio"] < 0.3:
            return "Downgrade Plan"
        else:
            return "Keep Current Plan"
    return "Admin"

# ------------------------------
# Cluster-based recommendation
# ------------------------------
def cluster_recommendation(df: pd.DataFrame):
    if df.empty:
        return df
    n_clusters = min(3, len(df))
    if n_clusters < 1:
        df["ClusterReco"] = "Not enough data for clustering"
        return df
    features = df[["data_used", "plan_price"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_plan = df.groupby("Cluster")["data_quota"].agg(lambda x: x.value_counts().index[0])

    def cluster_based(row):
        X_new = pd.DataFrame([[row["data_used"], row["plan_price"]]], columns=["data_used", "plan_price"])
        cluster = kmeans.predict(scaler.transform(X_new))[0]
        return f"Users like you prefer plan with {cluster_plan[cluster]} GB quota"

    df["ClusterReco"] = df.apply(cluster_based, axis=1)
    return df

# ------------------------------
# USER Endpoint
# ------------------------------
@app.get("/recommendations/{user_id}")
async def user_recommendation(user_id: str):
    df = fetch_subscriptions()
    if df is None or df.empty:
        return {"error": "No subscription data"}

    user = df[df["user_id"] == user_id]
    if user.empty:
        return {"error": "User not found"}

    user["Recommendation"] = user.apply(rule_based_recommendation, axis=1)
    user = cluster_recommendation(user)
    user = add_churn_prediction(user)

    today = pd.Timestamp(datetime.now())
    user["RenewalDue"] = user["end_date"].apply(
        lambda x: (pd.to_datetime(x) - today).days if pd.notnull(x) else None
    )

    return user[[
        "user_id", "name", "role", "plan_name", "data_quota", "data_used",
        "UsageRatio", "Recommendation", "ClusterReco", "ChurnPrediction", "RenewalDue"
    ]].to_dict(orient="records")[0]

# ------------------------------
# ADMIN Endpoint
# ------------------------------
@app.get("/admin/recommendations")
async def admin_recommendations():
    df = fetch_subscriptions()
    if df is None or df.empty:
        return {"error": "No subscription data"}

    df["Recommendation"] = df.apply(rule_based_recommendation, axis=1)
    df = cluster_recommendation(df)
    df = add_churn_prediction(df)

    today = pd.Timestamp(datetime.now())
    df["RenewalDue"] = df["end_date"].apply(
        lambda x: (pd.to_datetime(x) - today).days if pd.notnull(x) else None
    )
    df["RenewalAlert"] = df["RenewalDue"].apply(lambda x: x <= 7 if x is not None else False)

    plan_summary = df.groupby("plan_name")["user_id"].count().to_dict()
    avg_usage = round(df["UsageRatio"].mean(), 2)
    renewals_due = df[df["RenewalAlert"] == True][
        ["user_id", "name", "plan_name", "RenewalDue"]
    ].to_dict(orient="records")

    return {
        "users": df[[
            "user_id", "name", "role", "plan_name", "data_quota", "data_used",
            "UsageRatio", "Recommendation", "ClusterReco", "ChurnPrediction", "RenewalDue"
        ]].to_dict(orient="records"),
        "plan_summary": plan_summary,
        "average_usage_ratio": avg_usage,
        "renewals_due_soon": renewals_due
    }


# ------------------------------
# ANALYTICS SUMMARY Endpoint
# ------------------------------
@app.get("/analytics/summary")
async def analytics_summary():
    df = fetch_subscriptions()
    if df is None or df.empty:
        return {"error": "No subscription data"}
    # Use 'role' as status and 'data_used' as usage (GB)
    status_counts = df['role'].value_counts().to_dict()
    avg_usage = round(df['data_used'].mean(), 2)
    return {
        "status_distribution": status_counts,
        "average_usage_gb": avg_usage
    }

# ------------------------------
# ROOT
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Subscription Analytics API is running 🚀"}
