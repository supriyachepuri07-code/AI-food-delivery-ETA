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
@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the Data Validation component.
    """

    train_data_path: Path
    test_data_path: Path
@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the Data Transformation component.
    """

    train_data_path: Path
    test_data_path: Path
    preprocessor_pbject_file_path: Path