from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', BASE_DIR / 'data' / 'raw'))
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
PREDICTIONS_DIR = BASE_DIR / 'predictions'
POWERBI_EXPORT_PATH = Path(os.getenv('POWERBI_EXPORT_PATH', BASE_DIR / 'exports'))
import os
from dotenv import load_dotenv
load_dotenv()  # Charge les variables depuis .env

POWERBI_EXPORT_PATH='./exports'
from dotenv import load_dotenv
import os



load_dotenv()  # Charge les variables depuis .env

# Exemple : accéder à une variable
raw_data_dir = os.getenv("RAW_DATA_DIR")
print(f"Raw data directory: {raw_data_dir}")