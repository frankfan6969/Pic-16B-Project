# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:52:11 2023

@author: Gavin Joyce
"""

from LA_crime_predictor import crime_db as cdb
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns

def crime_count_year(year_begin,year_end):
    '''
    Takes in a start year and end year for the query function, performs the database query,
    then creates and displays a plot of the crime count by year.

    Parameters
    ----------
    year_begin : int
        first year the user would like to query
    year_end : int
        last year the user would like to query

    Returns
    -------
    None.

    '''
    df = cdb.query_years(year_begin, year_end)
    # creates a new grouped dataframe for counting purposes
    crime_num = df.groupby("year")["Vict Age"].agg(len).reset_index() 
    crime_num.rename(columns = {"Vict Age" : "Crime Count"}, inplace = True)
    
    # here we set the x and y axes of our plot
    plt.bar(crime_num["year"], crime_num["Crime Count"])
    # here we set labels and a title for our plot
    plt.xlabel("Year")
    plt.ylabel("Number of Crime Instances")
    plt.title(f"Crime Instances Reported by Year, {year_begin}-{year_end}")
    plt.show()
    
def crime_count_period(year_begin,year_end):
    '''
    Takes in a start year and end year for the query function, performs the database query,
    then creates and displays a plot of the crime count by crime period and year.

    Parameters
    ----------
    year_begin : int
        first year the user would like to query
    year_end : int
        last year the user would like to query

    Returns
    -------
    None.

    '''
    df = cdb.query_years(year_begin, year_end)
    
    # here we set the x-axis of our plot and set each bar's color to correspond to a crime period
    sns.countplot(df, x="year", hue="Crime Period")
    plt.title(f"Crime Count by Crime Period and Year, {year_begin}-{year_end}")
    # place legend outside top right corner of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    
def crime_count_age(year_begin,year_end):
    '''
    Takes in a start year and end year for the query function, performs the database query,
    then creates and displays a plot of the crime count by victim age group and year.

    Parameters
    ----------
    year_begin : int
        first year the user would like to query
    year_end : int
        last year the user would like to query

    Returns
    -------
    None.

    '''
    df = cdb.query_years(year_begin, year_end)
    # creates a new grouped dataframe for counting purposes
    agegroup_num = df.groupby(["year","Vict Age Group"])["DATE OCC"].agg(len).reset_index()
    agegroup_num.rename(columns = {"DATE OCC" : "Victim Count"}, inplace = True)
    
    # here we define the x and y axes of our plot and set a portion of each bar to a color
    # corresponding to a victim age group; we also set a title
    fig = px.bar(agegroup_num,x="year",y="Victim Count", color = "Vict Age Group", 
             title = f"Crime Count by Victim Age Group and Year, {year_begin}-{year_end}")
    fig.show()
    
def crime_count_sex(year_begin,year_end):
    '''
    Takes in a start year and end year for the query function, performs the database query,
    then creates and displays a plot of the crime count by victim sex and year.

    Parameters
    ----------
    year_begin : int
        first year the user would like to query
    year_end : int
        last year the user would like to query

    Returns
    -------
    None.

    '''
    df = cdb.query_years(year_begin, year_end)
    
    # here we define the x-axis of our plot and set each bar to have a color corresponding to a victim sex
    sns.countplot(df, x="year", hue="Vict Sex").set(title=f"Crime Count by Victim Sex and Year, {year_begin}-{year_end}")
    # place legend outside top right corner of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

def crime_map(year_begin,year_end):
    '''
    Takes in a start year and end year for the query function, performs the database query,
    then creates and displays a mapbox plot showing crime locations, colored by crime period.
    Hovering over the points also shows a description of each crime.

    Parameters
    ----------
    year_begin : int
        first year the user would like to query
    year_end : int
        last year the user would like to query

    Returns
    -------
    None.

    '''
    df = cdb.query_years(year_begin, year_end)
    # this gives us access to mapbox 
    px.set_mapbox_access_token("pk.eyJ1IjoiZ2pveWNlODA1IiwiYSI6ImNsbzF2cWYydzFsa24yaW82OGFiNDA3MDUifQ.gBGJPQQphfnWPWTaY4LqwA")
    
    # here we define the latitude and longitude columns for our plot, as well as  
    # hover information and color for each point, and a title
    fig = px.scatter_mapbox(df, lat = "LAT",lon = "LON", hover_name = "Crm Cd Desc", color = "Crime Period",
                            title = f"Map of Crime Instances, {year_begin}-{year_end}")
    fig.show()