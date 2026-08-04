import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.logger import logging
from src.utils.exception import CustomException

from src.entity.config_entity import DataIngestionConfig
from src.config.configuration import ConfigurationManager