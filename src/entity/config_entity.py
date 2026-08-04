from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    """

    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path