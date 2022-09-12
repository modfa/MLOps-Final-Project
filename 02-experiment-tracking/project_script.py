# =============================================================================
# PREDICTING PRICE OF PRE-OWNED CARS 
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import argparse
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


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



def run(raw_data_path: str, dest_path: str, dataset: str = "cars_sampled.csv"):
    X_train, X_test, y_train, y_test = read_Dataframe(
        os.path.join(raw_data_path, dataset))
    
    dv = DictVectorizer()
    X_train, dv = preprocess(X_train, dv, fit_dv=True)
    X_test, _ = preprocess(X_test, dv, fit_dv=False)
    
    
     # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save dictvectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
#     dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="the location where the raw NYC taxi trip data was saved"
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)

    