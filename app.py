from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv
import pandas as pd
import plotly.plotly
import plotly.offline as offline
import plotly.graph_objs as go
import seaborn as sns
import os
from os import environ
import shutil
import argparse
from flask import Flask, render_template
import json

app = Flask(__name__)

def plot():
    data = pd.read_csv("wine-reviews/winemag-data_first150k.csv")
    temp = data["points"].value_counts()
    data = [go.Bar(
            x=temp.index,
            y=temp.values,
            marker=dict(
            color='purple'
            )
    )]
    layout = go.Layout(
        autosize=True,
        title="feature selection"
)

    figure = go.Figure(data=data, layout=layout)
    return offline.plot(figure)


@app.route('/')
def index():
    return plot()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 9999))
    app.run(host='0.0.0.0', port=port)
