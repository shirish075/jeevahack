from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_excel("SubscriptionUseCase_Dataset.xlsx")

# ------------------------------
# Generate Synthetic Data (since dataset only has basic info)
# ------------------------------
np.random.seed(42)

df['Quota'] = np.random.choice([50, 100, 150, 200], size=len(df))
df['Usage'] = (df['Quota'] * np.random.uniform(0, 1.2, size=len(df))).astype(int)
plan_price_map = {50: 299, 100: 499, 150: 799, 200: 999}
df['PlanPrice'] = df['Quota'].map(plan_price_map)
df['UsageRatio'] = df['Usage'] / df['Quota']

# ------------------------------
# Rule-based Recommendation
# ------------------------------
def rule_based_recommendation(row):
    status = str(row['Status']).lower()
    if status == "active":
        if row['UsageRatio'] > 0.8:
            return "Upgrade Plan"
        elif row['UsageRatio'] < 0.3:
            return "Downgrade Plan"
        else:
            return "Keep Current Plan"
    elif status == "inactive":
        return "Send Discount / Re-Engage"
    else:
        return "Monitor Usage"

df['Recommendation'] = df.apply(rule_based_recommendation, axis=1)

# ------------------------------
# ML-based Clustering Recommendation
# ------------------------------
features = df[['Usage', 'PlanPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_plan = df.groupby('Cluster')['Quota'].agg(lambda x: x.value_counts().index[0])

def cluster_based_recommendation(row):
    X_new = pd.DataFrame([[row['Usage'], row['PlanPrice']]], columns=['Usage', 'PlanPrice'])
    cluster = kmeans.predict(scaler.transform(X_new))[0]
    return f"Users like you prefer Plan with {cluster_plan[cluster]} GB quota"

df['ClusterReco'] = df.apply(cluster_based_recommendation, axis=1)

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="Analytics Engine API")

@app.get("/")
async def root():
    return {"message": "Analytics Engine is running 🚀"}

@app.get("/recommendations/{user_id}")
async def get_user_recommendation(user_id: int):
    user = df[df['User Id'] == user_id]
    if user.empty:
        return {"error": "User not found"}
    return user[['User Id', 'Name', 'Status', 'Usage', 'Quota', 'PlanPrice',
                 'Recommendation', 'ClusterReco']].to_dict(orient="records")[0]

@app.get("/recommendations")
async def get_all_recommendations():
    return df[['User Id', 'Name', 'Status', 'Usage', 'Quota', 'PlanPrice',
               'Recommendation', 'ClusterReco']].to_dict(orient="records")

@app.get("/analytics/summary")
async def analytics_summary():
    status_counts = df['Status'].value_counts().to_dict()
    avg_usage = round(df['Usage'].mean(), 2)
    return {
        "status_distribution": status_counts,
        "average_usage_gb": avg_usage
    }
