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

application = Flask(__name__)
@application.route('/')
def index():

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

    figure = go.Figure(data=data, layout=layout) # make the templates dir
    new_path = '/opt/app-root/src/templates'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        # move the file to the templates dir
        shutil.move('/opt/app-root/src/temp-plot.html', new_path)
    return render_template("temp-plot.html", title='Plot')


if __name__ == '__main__':
    app.run()
