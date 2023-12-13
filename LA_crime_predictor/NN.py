# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:09:42 2023

@author: Jiayi Fan
"""

# imports
from LA_crime_predictor import crime_db as cd
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder



#Global Variables
le_time = LabelEncoder()
le_VictAge = LabelEncoder()
le_risk = LabelEncoder()
scalars = ["Crime Period", "Vict Age Group", "LAT", "LON"]

scalars_input = keras.Input(
    shape = (len(scalars), ),
    name = "scalars",
    dtype = "float64"
)


def Encode_Df(df):
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
    #X= df[["Crime Period", "Vict Age Group", "LAT", "LON", "Risk"]]
    df["Crime Period"] = le_time.fit_transform(df["Crime Period"])
    df["Vict Age Group"] = le_VictAge.fit_transform(df["Vict Age Group"])
    df["Risk"] = le_risk.fit_transform(df["Risk"])
    return df


def Make_Data(df):
    '''
    Takes in a pandas dataframe and convert it to TensorFlow Dataset

    Parameters
    ----------
    df: pandas dataframe

    Returns
    -------
    a TensorFlow dataset
    '''

    data = tf.data.Dataset.from_tensor_slices(
        (
            #they wann be in the same model --> group
            {
                "scalars" : df[scalars]
            },
            {
                "risk" : df[["Risk"]]
            }
        )
    )

    return data



def train_model(year_begin, year_end, target_month):
    '''
    Takes in a beginning year and end year as integers, opens a database connection,
    returns a trained NN model that fits the data.
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
    (Trained NN model, training history of the model)
    '''
    df = cd.query_years(year_begin, year_end)
    #get the month information and extract all the data that match the target_month
    df["Month"] = df["DATE OCC"].str[0:2].astype(int)
    train_df = df[df["Month"] == target_month]
    #transform categorical variables to numericals ones
    filt_train_df = Encode_Df(train_df)
    data = Make_Data(filt_train_df)

    data = data.shuffle(buffer_size = len(data), reshuffle_each_iteration=False)

    train_size = int(0.9*len(data))
    val_size   = int(0.1*len(data))
    #run SGD on sample of 100
    train = data.take(train_size).batch(50)
    val   = data.skip(train_size).take(val_size).batch(50)

    
    #Define Layers of the Model
    scalar_features = layers.Reshape((len(scalars), 1), input_shape=(len(scalars),))(scalars_input)

    scalar_features = layers.Conv1D(filters = 18, kernel_size=3, activation='relu')(scalar_features)
    scalar_features = layers.BatchNormalization()(scalar_features)
    scalar_features = layers.MaxPooling1D(pool_size = 2, strides = 1, padding = "valid")(scalar_features)
    scalar_features = layers.Dropout(0.2)(scalar_features)


    scalar_features = layers.LSTM(32, return_sequences=True)(scalar_features)
    scalar_features = layers.LSTM(16)(scalar_features)

    scalar_features = layers.Dense(64, activation='relu')(scalar_features)
    scalar_features = layers.BatchNormalization()(scalar_features)
    scalar_features = layers.Dropout(0.2)(scalar_features)

    scalar_features = layers.Dense(3, activation='softmax', name = 'risk')(scalar_features)
    output = scalar_features

    #Establish the Model
    model = keras.Model(
        inputs = scalars_input,
        outputs = output
    )
    #Compile the Model
    model.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
    )
    #Fit the Model
    history = model.fit(train,
                    validation_data=val,
                    epochs = 50,
                    verbose = True)
    
    return model, history 




def test_model(clf, target_year, target_month):
    '''
    Parameters
    ----------
    clf: a NN model that you just finished training by using train_model function
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

    #transform categorical variables to numericals ones and obtain the test data
    filt_test_df = Encode_Df(test_df)
    test_data = Make_Data(filt_test_df)
    #batch the test data
    test = test_data.batch(100)
    return clf.evaluate(test)