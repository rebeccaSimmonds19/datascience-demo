from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import environ
import shutil
import argparse
from flask import Flask, render_template

app = Flask(__name__)

data = pd.read_csv("wine-reviews/winemag-data_first150k.csv")

def pastel_plot(df, x, y):
    plt.figure(figsize = (15,6))
    plt.title('Points histogram - whole dataset')
    sns.set_color_codes("pastel")
    sns.barplot(x = x, y=y, data=data)
    locs, labels = plt.xticks()
    plt.show()

temp = data["points"].value_counts()
pastel_plot(data,temp.index, temp.values)

def make_template():
    # make the templates dir
    new_path = '/opt/app-root/src/templates'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        # move the file to the templates dir
        shutil.move('/opt/app-root/src/map.html', new_path)
    return render_template("features.html", title='Feature selection')

@app.route('/')
def index():
    return make_template()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
