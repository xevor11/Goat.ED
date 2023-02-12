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

def get_fighter_url(fighter):
    buildargs = {
        "serviceName": "customsearch",
        "version": "v1",
        "developerKey": GOOGLE_API_DEVELOPER_KEY,
    }

    cseargs = {
        "q": fighter + " " + "Official Fighter Profile",
        "cx": CSE_ID,
        "num": 1,
        "imgSize": "large",
        "searchType": "image",
        "fileType": "png",
        "safe": "off",
    }

    cseargs2 = {
        "q": "Not found",
        "cx": "015743272077172593992:pfwmvxoiylc",
        "num": 1,
        "imgSize": "large",
        "searchType": "image",
        "fileType": "png",
        "safe": "off",
    }

    results = search_google.api.results(buildargs, cseargs)
    url = results.links[0]

    if url:
        return url
    else:
        return (
            "https://pngimage.net/wp-content/uploads/2018/06/image-not-found-png-9.png"
        )

colors = {"background": "#FAFBFC", "text": "#34495E"}

size = {"font": "20px"}

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.scripts.config.serve_locally = True

app.layout = html.Div(
    style={"backgroundColor": colors["background"], "height": "650px"},
    children=[
        html.H1("UFC Predictions", style={"textAlign": "center"}),
        html.Div(
            style={"textAlign": "center"},
            children=[
                html.Div(
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "textAlign": "center",
                    },
                    children=[
                        html.Label(
                            "Select Weightclass", style={"fontSize": size["font"]}
                        ),
                        dcc.Dropdown(
                            id="weightclass",
                            options=[
                                {"label": wt.upper(), "value": wt}
                                for wt in df_weight_classes.keys()
                            ],
                            value="Lightweight",
                        ),
                        html.Br(),
                        html.Label(
                            "Number of Rounds", style={"fontSize": size["font"]}
                        ),
                        dcc.Dropdown(
                            id="no_of_rounds",
                            options=[
                                {"label": "5 Rounds", "value": 5},
                                {"label": "3 Rounds", "value": 3},
                            ],
                            value=3,
                        ),
                        html.Br(),
                        html.Label("Fight type", style={"fontSize": size["font"]}),
                        dcc.Dropdown(id="fight_type", value="Non Title"),
                    ],
                ),
                html.Div(
                    style={"width": "30%", "float": "left", "textAlign": "left"},
                    children=[
                        html.Label(
                            "Red Corner (Favourite)",
                            style={"textAlign": "center", "fontSize": "35px"},
                        ),
                        html.Label("Select Fighter", style={"fontSize": size["font"]}),
                        dcc.Dropdown(id="red-fighter"),
                        html.Br(),
                        html.Center(html.Img(id="red-image", width="100%")),
                    ],
                ),
                html.Div(
                    style={"width": "30%", "float": "right", "textAlign": "left"},
                    children=[
                        html.Label(
                            "Blue Corner (Underdog)",
                            style={"textAlign": "center", "fontSize": "35px"},
                        ),
                        html.Label("Select Fighter", style={"fontSize": size["font"]}),
                        dcc.Dropdown(id="blue-fighter"),
                        html.Br(),
                        html.Center(html.Img(id="blue-image", width="100%")),
                    ],
                ),
                html.Div(
                    style={
                        "width": "40%",
                        "marginLeft": "auto",
                        "marginRight": "auto",
                        "textAlign": "left",
                    },
                    children=[
                        html.Br(),
                        html.Br(),
                        html.Center(
                            html.Button(
                                "Predict",
                                id="button",
                                style={
                                    "fontSize": "32px",
                                    "backgroundColor": "rgba(255,255,255,0.8)",
                                },
                            )
                        ),
                        html.Br(),
                        html.Div(
                            style={
                                "width": "35%",
                                "float": "left",
                                "textAlign": "left",
                                "backgroundColor": "rgba(255,255,255,0.7)",
                            },
                            children=[
                                html.H2(
                                    "Red-Corner",
                                    style={
                                        "textAlign": "center",
                                        "color": "rgb(102, 0, 0)",
                                    },
                                ),
                                html.H3(
                                    children=["click \n predict"],
                                    id="red-proba",
                                    style={"textAlign": "center"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "35%",
                                "float": "right",
                                "textAlign": "left",
                                "backgroundColor": "rgba(255,255,255,0.7)",
                            },
                            children=[
                                html.H2(
                                    "Blue-Corner",
                                    style={
                                        "textAlign": "center",
                                        "color": "rgb(0, 51, 102)",
                                    },
                                ),
                                html.H3(
                                    children=["click \n predict"],
                                    id="blue-proba",
                                    style={"textAlign": "center"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                dcc.Markdown(
                    """
                #### Web App by [Rajeev Warrier](https://rajeevwarrier.com)
                **Disclaimer**: I do not condone this app's use for betting. I am not responsible for
                any damage done or losses incurred by way of this app.
                """.replace(
                        "  ", ""
                    )
                )
            ],
            style={"text-align": "center", "margin-bottom": "15px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    """
                #### Details

                **AUC score: 72.20%**

                The app does not require any odds. It works purely on stats.

                You can read more on my github [repo](https://github.com/WarrierRajeev/UFC-Predictions)

                You can read more about me and contact me from my [website](https://rajeevwarrier.com/)


                ##### Helpful UFC links

                You can find upcoming events [here](https://www.ufc.com/events)

                You can find all the athletes competing [here](https://www.ufc.com/athletes/all)



                """.replace(
                        "  ", ""
                    )
                )
            ],
            style={"text-align": "left", "margin-bottom": "15px"},
        ),
        html.Br(),
        html.Br(),
    ],
)


