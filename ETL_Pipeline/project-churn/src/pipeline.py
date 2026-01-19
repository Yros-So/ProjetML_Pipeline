from prefect import flow
from pathlib import Path
import pandas as pd

from src.etl import run_etl
from src.train import train_and_save_best_model
from src.predict import predict

@flow
def churn_pipeline():

    raw_dir = Path("data/raw")

    processed = run_etl(raw_dir)
    train_and_save_best_model(str(processed))

    df = pd.read_csv(processed)
    df_pred = predict(df)

    out = Path("data/predictions/predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out, index=False)

if __name__ == "__main__":
    churn_pipeline()
