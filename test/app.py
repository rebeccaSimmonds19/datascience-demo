from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
import seaborn as sns
import os
from os import environ
import shutil
import argparse
from flask import Flask, render_template

app = Flask(__name__)

data = pd.read_csv("wine-reviews/winemag-data_first150k.csv")

@app.route('/')
def bar_plot(x, y):
    data = [go.Bar(
            x=x,
            y=y,
            marker=dict(
            color='purple'
            )
    )]
    layout = go.Layout(
        autosize=True,
        title="feature selection"

    )
    #sns.barplot(x = x, y=y, data=data)
    #plt.show(filename="feature.html")
    fig = go.Figure(data=data, layout=layout)
    print(offline.plot(fig, filename='features.html'))

def index():
    temp = data["points"].value_counts()
    bar_plot(temp.index, temp.values)
