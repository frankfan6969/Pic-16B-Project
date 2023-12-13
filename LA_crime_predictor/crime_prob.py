# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:54:07 2023

@author: Gavin Joyce
"""

from LA_crime_predictor import crime_db as cdb
from scipy.stats import poisson,expon
import numpy as np
import matplotlib.pyplot as plt


def calc_lambda(address):
    '''
    Takes in an address as a string, calls the query_address function with the address
    as a parameter, counts the number of crimes in the query, and then calculates lambda
    for the query.

    Parameters
    ----------
    address : str
        The street address the user would like to search around for crimes

    Returns
    -------
    l : float
        short for lambda, the average number of crimes occurring in the area per day,
        which is also the parameter for the poisson and exponential distributions

    '''
    df = cdb.query_address(address)
    num_crimes = df.agg(len)[0] # counts number of rows in the dataframe
    num_days = 2 * 365 
    # here we count the number of days in 2 years since our query only looks at 
    # the last 2 complete years
    l = round(num_crimes/num_days, 4)
    
    return l

def plot_poisson(address):
    '''
    Takes in an address as a string, calls the calc_lambda function with the address as
    the parameter, creates a numpy array for the x-axis and then a y-value for each x according
    to the Poisson probability distribution. Finally, plots the x and y values using 
    matplotlib and adds labels to the plot.

    Parameters
    ----------
    address : str
        The street address the user would like to search around for crimes

    Returns
    -------
    None.

    '''
    l = calc_lambda(address)
    x = np.arange(0, (10*l), 1) # since the poisson distribution is discrete, step size of 1 makes the most sense
    y = poisson.pmf(x, mu=l) # here we specify lambda as the mean of the Poisson distribution
    # for sufficiently large λ, the poisson distribution approximates the normal distribution
    
    plt.plot(x, y)
    plt.xlabel("# of Crimes Today")
    plt.ylabel("Probability")
    new_line = "\n"
    plt.title(f"Poisson Distribution of Crime Occurrences {new_line}(λ = {l}, Address: {address})")
    plt.show()
    
def plot_expon(address):
    '''
    Takes in an address as a string, calls the calc_lambda function with the address as
    the parameter, creates a numpy array for the x-axis and then a y-value for each x according
    to the exponential probability distribution. Finally, plots the x and y values using 
    matplotlib and adds labels to the plot.

    Parameters
    ----------
    address : str
        The street address the user would like to search around for crimes

    Returns
    -------
    None.

    '''
    l = calc_lambda(address)
    l_inverse = round(1/l,4)
    # since the exponential distribution is continuous, we can use a much smaller step size
    x = np.arange(0, 5, 0.01) 
    # here we specify 1/lambda as the mean of the expoenetial distribution
    y = expon.pdf(x, 0, scale=(l_inverse)) 
    
    plt.plot(x, y)
    plt.xlabel("Days Between Crime Occurrences")
    plt.ylabel("Probability")
    new_line = "\n"
    plt.title(f"Exponential Distribution of Days Between Crime Occurrences {new_line}(1/λ = {l_inverse}, Address: {address})")
    plt.show()