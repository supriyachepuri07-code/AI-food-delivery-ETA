from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)


class ConfigurationManager:
    """
    Responsible for creating configuration objects
    used by different pipeline components.
    """

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        ingestion_config = DataIngestionConfig(
            raw_data_path=RAW_DATA_DIR / "food_delivery.csv",
            train_data_path=PROCESSED_DATA_DIR / "train.csv",
            test_data_path=PROCESSED_DATA_DIR / "test.csv"
        )

        return ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:

        validation_config = DataValidationConfig(
            train_data_path=PROCESSED_DATA_DIR / "train.csv",
            test_data_path=PROCESSED_DATA_DIR / "test.csv"
        )

        return validation_config