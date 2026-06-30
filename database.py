import os
import pandas as pd
from sqlalchemy import create_engine

# Pull database credentials safely from system secret vaults
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Ensure modern parsing syntax compatibility for SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    engine = None

def load_combined_data(base_dataset_path="final_master_hackathon_dataset.csv"):
    df_base = pd.read_csv(base_dataset_path)
    if engine is None:
        return df_base
    try:
        df_user = pd.read_sql("SELECT * FROM user_learned_data", engine)
        df_user = df_user.drop(columns=['id', 'created_at'], errors='ignore')
        return pd.concat([df_base, df_user], ignore_index=True).dropna()
    except Exception:
        return df_base

def save_new_telemetry_to_cloud(data_dict):
    if engine is None:
        return False
    df_new = pd.DataFrame([data_dict])
    try:
        df_new.to_sql('user_learned_data', engine, if_exists='append', index=False)
        return True
    except Exception as e:
        print(f"Database write warning: {e}")
        return False
