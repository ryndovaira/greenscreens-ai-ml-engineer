from pathlib import Path
import pandas as pd

root_dir = Path.cwd().parent  # Assuming script is in root/EDA/
data_dir = root_dir / "dataset"
eda_dir = root_dir / "EDA"
eda_data_dir = eda_dir / "dataset"

train_path = data_dir / "train.csv"
validation_path = data_dir / "validation.csv"
test_path = data_dir / "test.csv"
eda_train_path = eda_data_dir / "train.csv"

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)
eda_train_df = pd.read_csv(train_path)