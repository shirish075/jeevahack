from supabase import create_client
import pandas as pd

SUPABASE_URL = "https://yzjvfkazpqevuxxmtjsf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl6anZma2F6cHFldnV4eG10anNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc3NDk5OTQsImV4cCI6MjA3MzMyNTk5NH0.c2bc6MU8UVleG8yY7BrGw6FAKPSXCfPBK--VjASxt-A"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_test_rows():
    response = supabase.table("subscriptions").select("*").limit(5).execute()
    
    # Check if data is present
    if not response.data:
        print("Error fetching data:", response)
        return pd.DataFrame()
    
    data = response.data  # this is a list of dicts
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = fetch_test_rows()
    print(df)
