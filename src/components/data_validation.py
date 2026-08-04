import os
import sys

import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException

from src.entity.config_entity import DataValidationConfig
from src.config.configuration import ConfigurationManager