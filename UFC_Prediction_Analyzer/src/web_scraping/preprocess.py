import math
import numpy as np
import pandas as pd

from src.createdata.preprocess_fighter_data import FighterDetailProcessor

from src.createdata.data_files_path import (  # isort:skip
    FIGHTER_DETAILS,
    PREPROCESSED_DATA,
    TOTAL_EVENT_AND_FIGHTS,
    UFC_DATA,
)
