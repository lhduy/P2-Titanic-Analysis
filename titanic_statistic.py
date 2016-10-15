import math
import numpy as np
import pandas as pd
import scipy.stats as sp

#function to find the linear correction (Pearson's correlation)
def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.
    
    correlation = average of (x in standard units) times (y in standard units)
    
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''
    std_x = (x-x.mean())/x.std(ddof=0)
    std_y = (y-y.mean())/y.std(ddof=0)
    
    return (std_x*std_y).mean()

#function to find the significantly different between sample and population means (Z-test)
def ztest_sample_population(x,u):
    meanx = x.mean()*1.0 #mean of sample data
    meanu = u.mean()*1.0 # mean of population data
    SD = u.std() #standard deviation of population
    nsample = x.count() # number of sample
    zstat = (meanx - meanu)/(SD/math.sqrt(nsample))
    pvalue = sp.norm.sf(abs(zstat)) #for one tailed testing
    return zstat,pvalue

def compute_freq_chi2(x,y):
    """This function will compute frequency table of x an y
    Pandas Series, and use the table to feed for the contigency table
    
    Parameters:
    -------
    x,y : Pandas Series, must be same shape for frequency table
    
    Return:
    -------
    None. But prints out frequency table, chi2 test statistic, and 
    p-value
    """
    freqtab = pd.crosstab(x,y)
    print("Frequency table")
    print("----------------------------")
    print(freqtab)
    print("----------------------------")
    chi2,pval,dof,expected = sp.chi2_contingency(freqtab)
    print("ChiSquare test statistic: ",chi2)
    print("p-value: ",pval)
    return