from pathlib import Path

data_version = "resolution=5min"
ROOT_DIR = Path(__file__).resolve().parent.parent

# change this according to your specific setup
DATA_DIR = Path("/Users/ntdoan/Downloads/data_set")

PROCESSED_DATA_DIR = ROOT_DIR / "data/processed" / data_version
TRAIN_FILE_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE_PATH = PROCESSED_DATA_DIR / "test.csv"
PAYLOAD_FILE_PATH = PROCESSED_DATA_DIR / "payload.json"
