import math
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import dash
import dash_core_components as dcc
import dash_html_components as html
import search_google.api
from dash.dependencies import Input, Output, State

GOOGLE_API_DEVELOPER_KEY = "enter_key_here"
CSE_ID = "enter_id_here"

fighter_df = pd.read_csv("app_data/latest_fighter_stats.csv", index_col="index")
weight_classes = pd.read_csv("app_data/weight_classes.csv")

with open("app_data/model.sav", "rb") as mdl:
    model = pickle.load(mdl)

with open("app_data/cols.list", "rb") as c:
    cols = pickle.load(c)

with open("app_data/standard.scaler", "rb") as ss:
    scaler = pickle.load(ss)

df_weight_classes = {
    "Flyweight": "weight_class_Flyweight",
    "Bantamweight": "weight_class_Bantamweight",
    "Featherweight": "weight_class_Featherweight",
    "Lightweight": "weight_class_Lightweight",
    "Welterweight": "weight_class_Welterweight",
    "Middleweight": "weight_class_Middleweight",
    "Light Heavyweight": "weight_class_LightHeavyweight",
    "Heavyweight": "weight_class_Heavyweight",
    "Women's Strawweight": "weight_class_Women_Strawweight",
    "Women's Flyweight": "weight_class_Women_Flyweight",
    "Women's Bantamweight": "weight_class_Women_Bantamweight",
    "Women's Featherweight": "weight_class_Women_Featherweight",
    "Catch Weight": "weight_class_CatchWeight",
    "Open Weight": "weight_class_OpenWeight",
}

def normalize(df: pd.DataFrame, scaler) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.float, np.int])
    df[list(df_num.columns)] = scaler.transform(df[list(df_num.columns)])
    return df


def get_age(X):

    median_age = 29

    DOB = pd.to_datetime(X)
    today = pd.to_datetime("today")

    if pd.isnull(DOB):
        return median_age
    else:
        age = math.floor((today - DOB).days / 365.25)
        return age

