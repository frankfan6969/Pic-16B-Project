# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:07:42 2023

@author: Gavin Joyce
"""

# imports
import pandas as pd
import numpy as np
import sqlite3
from datetime import time
from geopy.geocoders import Nominatim

class _classify: #classification functions used in df preparation for creating the database

    def crime_period(x):
        '''
        Takes in time occurred as a string, uses the datetime library to classify the time 
        into one of four categories, and then returns the category name as a string.
        
        Parameters
        ----------
        x : str
            time that the crime occurred in hours:minutes:seconds format

        Returns
        -------
        str
            category name of the time occurred (for example: "afternoon")

        '''
        if x < time(12,0,0) and x >= time(5, 0, 0):
            return "morning"

        elif x < time(17, 0, 0) and x >= time(12, 0, 0):
            return "afternoon"
        
        elif x < time(21, 0, 0) and x >=  time(17, 0, 0):
            return "evening"
        
        #9pm - 4 am
        else:
            return "night"
    
    def victim_age_group(x):
        '''
        Takes in the crime victim's age as an integer, classifies it into an age group,
        and then returns the name of the age group as a string.

        Parameters
        ----------
        x : int
            the age of the victim

        Returns
        -------
        str
            category name of the victim's age (for example: "young adult")

        '''
        if x < 18 and x >= 0:
            return "child"

        elif x < 30 and x >= 18:
            return "young adult"
        
        elif x < 45 and x >=  30:
            return "adult"
        
        elif x <60 and x >= 45:
            return "older adult"
        #60+
        else:
            return "senior"
        
    def risk(x):
        '''
        Takes in the crime's code as an integer, classifies it based on severity,
        and then returns the risk level as a string.

        Parameters
        ----------
        x : int
            crime code (number corresponding to a unique crime description)

        Returns
        -------
        str
            risk level (lower-valued crime codes correspond to more dangerous/severe
                        crimes)

        '''
        if x < 300:
            return "Serious"
        
        elif x >= 300 and x < 500:
            return "Medium"
        
        else:
            return "Light"

def create_db():
    '''
    Creates a new database by opening a connection, reads csv files in as pandas dataframes, 
    cleans the data, feeds the data into the database, and finally closes the database
    connection.
    '''
    # opens the database connection
    conn = sqlite3.connect("LA Crime Database.db")
    
    # df preparation columns
    col = ["DATE OCC", "TIME OCC", "AREA NAME", "Crm Cd", "Crm Cd Desc", 
           "Vict Age", "Vict Sex", "LOCATION", "LAT", "LON"]

    def prepare_df(df): # data cleaning to assemble our database
        '''
        Takes in a dataframe, performs several data cleaning and transformation operations,
        and then returns the modified data frame.

        Parameters
        ----------
        df 
            raw data read in from a csv file using pandas

        Returns
        -------
        df
            modified data frame that was cleaned and appended to using our classification
            functions
        '''
        df = df[col]
        df.iloc[:,6] = df["Vict Sex"].map({"M": "Male", "F": "Female", "X" : "Unknown", "H": "Unknown", "-": "Unknown", np.nan: "Unknown"})
        df.insert(10,"year",df["DATE OCC"].str[6:10])
        a = df['TIME OCC'].apply(lambda x: str(x).zfill(4))
        df.iloc[:,1] = pd.to_datetime(a, format='%H%M').dt.time
        
        df.insert(2,"Crime Period",df["TIME OCC"].transform(_classify.crime_period))
        df.insert(7,"Vict Age Group",df["Vict Age"].transform(_classify.victim_age_group))
        df.insert(5,"Risk",df["Crm Cd"].transform(_classify.risk))
        
        return(df)

    # names of csv files that will be read into the database
    files = ["LA_crime_predictor/crime_2010.csv", "LA_crime_predictor/crime_2011.csv", 
             "LA_crime_predictor/crime_2012.csv", "LA_crime_predictor/crime_2013.csv", 
             "LA_crime_predictor/crime_2014.csv", "LA_crime_predictor/crime_2015.csv",
             "LA_crime_predictor/crime_2016.csv", "LA_crime_predictor/crime_2017.csv", 
             "LA_crime_predictor/crime_2018.csv", "LA_crime_predictor/crime_2019.csv", 
             "LA_crime_predictor/crime_2020.csv", "LA_crime_predictor/crime_2021.csv",
             "LA_crime_predictor/crime_2022.csv", "LA_crime_predictor/crime_2023.csv"]
    
    # reads the csv files into the database
    for file in files:
        df_iter = pd.read_csv(file, chunksize = 100000)
        for df in df_iter:
            df = prepare_df(df)
            df.to_sql("crimes", conn, if_exists = "append", index = False)
            
    # closes the database connection
    conn.close()

def query_address(address): # within one hundredth of a degree, lat/lon
    '''
    Takes in an address as a string, opens a database connection, converts the address 
    to its latitude and longitude coordinates using the geopy library, returns 
    a dataframe containing all crimes that occurred near the address from 2021-2022,
    and finally closes the database connection.

    Parameters
    ----------
    address : str
        address the user would like to search around

    Returns
    -------
    df
        dataframe of all crimes committed near the given address from 2021-2022
    '''
    
    conn = sqlite3.connect("LA Crime Database.db")
    
    loc = Nominatim(user_agent="Geopy Library")
    getLoc = loc.geocode(address)
    lat = getLoc.latitude
    lat_min, lat_max = (lat-.01), (lat +.01) # sets the latitude search range 
    lon = getLoc.longitude
    lon_min, lon_max = (lon-.01), (lon+.01) # sets the longitude search range
    
    year_begin = 2021 # we only want to use recent data for accurate predictions
    year_end = 2022
    
    cmd = \
    f"""
    SELECT *
    FROM crimes C
    WHERE C.year <= {year_end} AND C.year >= {year_begin} AND C.LAT >= {lat_min} AND C.LAT <= {lat_max} AND C.LON >= {lon_min} AND C.LON <= {lon_max}
    """
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    
    return df


def query_years(year_begin, year_end):
    '''
    Takes in a beginning year and end year as integers, opens a database connection,
    returns a dataframe of all crimes that occurred during and between the start and
    end years, then closes the databse connection.

    Parameters
    ----------
    year_begin : int
        the first year the user would like to query
    year_end : int
        the final year the user would like to query

    Returns
    -------
    df
        dataframe of all crimes committed during and between the given years
    '''
    
    conn = sqlite3.connect("LA Crime Database.db")
    
    cmd = \
    f"""
    SELECT *
    FROM crimes C
    WHERE C.year <= {year_end} AND C.year >= {year_begin}
    """
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    
    return df