"""
<Credit card approval prediction using supervised machine learning models>

Copyright (c) 2020
Licensed
Written by <Jiaqi Fan, Balaram Pothamsetty, Swapnil Sakorkar, Azhagapan Chitha>

Intro:
Banking industries received so many applications for credit card request.
Going through each request manually can be very time consuming, also prone to human errors.
However, if we can use the historical data to build a model which can shortlist the candidates
for approval that can be great. In this project, we have tried to find out the factors that
are most important for getting an approval for the credit card through the power of Data
Analysis and Machine Learning. We have achieved 86% of accuracy using random forest Alorigthm

"""

############ Function 1: Encode Category Data ############

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def labelTrans(dataf):
    """
    Going through each columns and checking the type is object
    if it is object, encode category data
    """
    for col in dataf:
        if dataf[col].dtypes == 'object':
            dataf[col] = le.fit_transform(dataf[col])
    return dataf



##### Function 2: plot a univariate distribution of observation #####

import seaborn as sns
import matplotlib.pyplot as plt

def plotDistPlot(col):
    """Flexibly plot a univariate distribution of observation"""
    sns.distplot(col)
    return plt.show()

