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
        
   
    def _one_hot_encode_win(self):

        self.fights = pd.concat(
            [self.fights, pd.get_dummies(self.fights["win_by"], prefix="win_by")],
            axis=1,
        )
        self.fights.drop(["win_by"], axis=1, inplace=True)

    def _get_fighters(self):

        red_fighters = self.fights["R_fighter"].value_counts().index
        blue_fighters = self.fights["B_fighter"].value_counts().index

        return list(set(red_fighters) | set(blue_fighters))
