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

    Class feature_selection:

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

            X = data.drop(columns=['points'])

            X=X.fillna(-1)
            print(X.columns)

            categorical_features_indices =[0,1,2,3,4,5,6,7,8,9]
            y=data['points']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,random_state=52)

        def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test):
            model = CatBoostRegressor(
                random_seed = 400,
                loss_function = 'RMSE',
                iterations=400,
            )

            trained_model = model.fit(
                X_train, y_train,
                cat_features = categorical_features_indices,
                eval_set=(X_valid, y_valid),
                verbose=False
            )
            print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))
            print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))
            return model

            model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)

            feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),
                    columns=['Feature','Score'])

                    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

        plt.rcParams["figure.figsize"] = (12,7)
        ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
        ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
        ax.set_xlabel('')

        rects = ax.patches

        labels = feature_score['Score'].round(2)

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

        plt.show(filename=features.html)

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
    feature_selection()
    return make_template()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
