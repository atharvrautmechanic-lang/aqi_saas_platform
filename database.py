import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Safely pull connection string from Environment Secrets
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("CRITICAL ERROR: DATABASE_URL environment variable is missing!")

# 2. Establish persistent cloud connection pool
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def load_combined_data(base_dataset_path="final_master_hackathon_dataset.csv"):
    """Loads historical CSV and blends it with live custom entries from PostgreSQL."""
    df_base = pd.read_csv(base_dataset_path)

    try:
        query = "SELECT * FROM user_learned_data"
        df_user = pd.read_sql(query, engine)
        # Strip metadata columns to keep input shapes perfectly aligned for ML
        df_user = df_user.drop(columns=['id', 'created_at'], errors='ignore')
        print(f"📡 Synced {len(df_user)} custom rows from cloud database.")
    except Exception as e:
        print(f"⚠️ Cloud database empty or unreachable ({e}). Defaulting to baseline.")
        df_user = pd.DataFrame()

    return pd.concat([df_base, df_user], ignore_index=True).dropna()


def save_new_telemetry_to_cloud(data_dict):
    """Appends a fresh matrix row into the Supabase/Neon table."""
    df_new = pd.DataFrame([data_dict])
    try:
        df_new.to_sql('user_learned_data', engine, if_exists='append', index=False)
        print("🚀 Telemetry uploaded successfully to cloud database!")
        return True
    except Exception as e:
        print(f"❌ Database write error: {e}")
        return False