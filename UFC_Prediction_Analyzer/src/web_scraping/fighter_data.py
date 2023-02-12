import re

import numpy as np
import pandas as pd
from tqdm import tqdm


class FighterDetailProcessor:
    def __init__(self, fights, fighter_details):
        self.fights = fights
        self.fighter_details = fighter_details
        self._one_hot_encode_win()
        self.temp_red_frame, self.temp_blue_frame = self._calculate_fighter_data()
        self._convert_height_reach_to_cms()
        self._convert_weight_to_pounds()
        self.frame = self._merge_frames()
        self._rename_columns()
