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

def plot():
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
    #sns.barplot(x = x, y=y, data=data)
    #plt.show(filename="feature.html")
    fig = go.Figure(data=data, layout=layout)
    print(plot(fig, filename='features.html'))

def make_template():
    # make the templates dir
    new_path = '/opt/app-root/src/templates'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        # move the file to the templates dir
        shutil.move('/opt/app-root/src/feature.html', new_path)
    return render_template("features.html", title='Feature selection')

@app.route('/')
def index():
    plot()
    return make_template()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
