import pandas as pd
import pickle
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("project-final-experiment")


def read_Dataframe(path_to_df):
    cars_data=pd.read_csv(path_to_df)
    cars=cars_data.copy()
    col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
    cars=cars.drop(columns=col, axis=1)
    cars.drop_duplicates(keep='first',inplace=True)
    cars = cars[
            (cars.yearOfRegistration <= 2018) 
          & (cars.yearOfRegistration >= 1950) 
          & (cars.price >= 100) 
          & (cars.price <= 150000) 
          & (cars.powerPS >= 10) 
          & (cars.powerPS <= 500)]
    cars['monthOfRegistration']/=12

    # Creating new varible Age by adding yearOfRegistration and monthOfRegistration
    cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
    cars['Age']=round(cars['Age'],2)

    cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)
    col=['seller','offerType','abtest']
    cars=cars.drop(columns=col, axis=1)
    cars_copy=cars.copy()
    cars_omit=cars.dropna(axis=0)
    x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
    y1 = cars_omit['price']
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
    y_train = np.log(y_train.values)
    y_test = np.log(y_test.values)
    
    return X_train, X_test, y_train, y_test
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    categorical = ['vehicleType', 'gearbox','model', 'fuelType', 'brand', 'notRepairedDamage']
    numerical = ['powerPS','kilometer','Age']

    df[categorical] = df[categorical].astype(str)
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def run(dataset: str = "cars_sampled.csv"):
    X_train, X_test, y_train, y_test = read_Dataframe(dataset)
    
    dv = DictVectorizer()
    X_train, dv = preprocess(X_train, dv, fit_dv=True)
    X_test, _ = preprocess(X_test, dv, fit_dv=False)

    return X_train, X_test, y_train, y_test, dv


def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=10,
                evals=[(valid, 'validation')],
                early_stopping_rounds=5
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return


def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        
        # train = xgb.DMatrix(X_train, label=y_train)
        # valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
           'max_depth': 11,
            'learning_rate': 0.11504139773734708,
            'reg_alpha': 0.03143119240248877,
            'reg_lambda': 0.0058914904219020325,
            'min_child_weight': 9.242463709505468,
            'objective': 'reg:linear',
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")




if __name__ == "__main__":
    X_train, X_test, y_train, y_test, dv = run()

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)
    train_model_search(train, valid, y_test)
    train_best_model(train, valid, y_test, dv)