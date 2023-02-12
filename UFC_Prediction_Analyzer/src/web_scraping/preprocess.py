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

class Preprocessor:
    def __init__(self):
        self.FIGHTER_DETAILS_PATH = FIGHTER_DETAILS
        self.TOTAL_EVENT_AND_FIGHTS_PATH = TOTAL_EVENT_AND_FIGHTS
        self.PREPROCESSED_DATA_PATH = PREPROCESSED_DATA
        self.UFC_DATA_PATH = UFC_DATA
        self.fights = None
        self.fighter_details = None
        self.store = None

    def process_raw_data(self):
        print("Reading Files")
        self.fights, self.fighter_details = self._read_files()

        print("Drop columns that contain information not yet occurred")
        self._drop_future_fighter_details_columns()

        print("Renaming Columns")
        self._rename_columns()
        self._replacing_winner_nans_draw()

        print("Converting Percentages to Fractions")
        self._convert_percentages_to_fractions()
        self._create_title_bout_feature()
        self._create_weight_classes()
        self._convert_last_round_to_seconds()
        self._convert_CTRL_to_seconds()
        self._get_total_time_fought()
        self.store = self._store_compiled_fighter_data_in_another_DF()
        self._create_winner_feature()
        self._create_fighter_attributes()
        self._create_fighter_age()
        self._save(filepath=self.UFC_DATA_PATH)

        print("Fill NaNs")
        self._fill_nas()
        print("Dropping Non Essential Columns")
        self._drop_non_essential_cols()
        self._save(filepath=self.PREPROCESSED_DATA_PATH)
        print("Successfully preprocessed and saved ufc data!\n")

