from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import yaml
from munch import munchify


# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

#### ADJUST FOR YOUR MACHINE ####
# TODO: Needs to be updated
DATA_ROOT = Path("/data_users1/sagar/winogender_contextuality")
logger.info(f"DATA_ROOT path is: {PROJ_ROOT}")

DATA_DIR = DATA_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = DATA_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Model Specifications
API_KEY = ''
GPU_INDEX = 'gpu:0'

# MODEL PARAMETERS
with open("params.yaml", "r") as f:
    doc = yaml.safe_load(f)
MODEL_PARAMS = munchify(doc)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
