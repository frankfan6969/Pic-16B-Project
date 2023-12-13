# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:09:42 2023

@author: Jiayi Fan
"""

# imports
from LA_crime_predictor import crime_db as cd
import pandas as pd
import numpy as np
from datetime import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#Global Variables
le_time = LabelEncoder()
le_VictAge = LabelEncoder()
le_risk = LabelEncoder()


def Encode_Input(df):
    '''
    Takes in an unfiltered dataframe with categorical entries as input
    Select specific entries from the dataset and make them all numerical

    Parameters
    ----------
    df: pandas dataframe

    Returns
    -------
    a filtered pandas dataframe with all entries numerical
    '''
    #select following columns as trained inputs
    X= df[["Crime Period", "Vict Age Group", "LAT", "LON"]]
    X["Crime Period"] = le_time.fit_transform(df["Crime Period"])
    X["Vict Age Group"] = le_VictAge.fit_transform(df["Vict Age Group"])
    return X


def Encode_Label(df):
    '''
    Takes in an unfiltered dataframe with categorical entries as input
    Select specific entries from the dataset and make them all numerical

    Parameters
    ----------
    df: pandas dataframe

    Returns
    -------
    a filtered pandas dataframe with all entries numerical
    '''

    y = df[["Risk"]]
    y["Risk"] = le_risk.fit_transform(y["Risk"])
    return y


def train_model(year_begin, year_end, target_month):
    '''
    Takes in a beginning year and end year as integers, opens a database connection,
    returns a trained model using Random Forest of depth 12 using data from this period.
    We will only look at data with the same month as the target_month.

    The Returned Model aims to predict the type of crime given "Crime Period", "Vict Age Group","LAT", "LON" as input

    Parameters
    ----------
    year_begin : int
        the first year the user would like to query
    year_end : int
        the final year the user would like to query
    target_month: int
        the month the user would like to investigate

    Returns
    -------
    (Trained Random Forest Model of Depth 12, training score on the data)
    '''
    df = cd.query_years(year_begin, year_end)
    #get the month information and extract all the data that match the target_month
    df["Month"] = df["DATE OCC"].str[0:2].astype(int)
    train_df = df[df["Month"] == target_month]
    #transform categorical variables to numericals ones
    X_train = Encode_Input(train_df)
    y_train = Encode_Label(train_df)

    #Establish our Model
    forest = RandomForestClassifier(max_depth=12)
    forest.fit(X_train, y_train)

    return forest, forest.score(X_train, y_train)



def test_model(clf, target_year, target_month):
    '''
    Parameters
    ----------
    clf: a ML model that you just finished training by using train_model function
    target_year: int, the year you would like your model to be tested on
    target_month: int, the month you would like your model to be tested on

    Note: target_month should be the same as the target month you trained the model on, 
    otherwise the result will not guaranteed to be indicative

    Returns
    -------
    accuracy score when applied the model to the test data
    '''

    df = cd.query_years(target_year, target_year)
    #get the month information and extract all the data that match the target_month
    df["Month"] = df["DATE OCC"].str[0:2].astype(int)
    test_df = df[df["Month"] == target_month]

    #transform categorical variables to numericals ones
    X_test = Encode_Input(test_df)
    y_test = Encode_Label(test_df)

    y_true = y_test["Risk"]
    y_pred = clf.predict(X_test)
    return accuracy_score(y_true, y_pred, normalize=True)

    

def predict_crime_type(clf, crime_period, age_group, LAT, LON):
    '''
    Parameters
    ----------
    clf: a ML model that you just finished training by using train_model function
    crime_period: string, the current period of the day the user is in
    age_group: int, the age group the user belongs to
    LAT: int, the current latitude of the user
    LON: int, the current longitude of the user

    Note: In Practice, We can get LAT and LON using GPS without manually input the data

    Returns
    -------
    the predicted crime type to the user if there is any
    '''

    data = {'Crime Period': [le_time.fit_transform(np.array([crime_period]))], 'Vict Age Group': [le_VictAge.fit_transform(np.array([age_group]))], 
            "LAT": [np.array([LAT])], "LON": [np.array([LON])]}
    
    X = pd.DataFrame.from_dict(data)
    y_pred = clf.predict(X)
    #map the y_pred to the crime_type using inverse_transform method
    return le_risk.inverse_transform(y_pred)

