# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:40:35 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

loansData = pd.read_csv('loansData.csv')

# Droping missing observations
loansData.dropna(inplace=True)

# Cleaning the variable Interest.Rate, removing percentage point. 
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate

# Setting Home.Ownership
#loansData['Home_Ownership_ord'] = pd.Categorical(loansData['Home.Ownership']).labels
# base case mortgate
dummies = pd.get_dummies(loansData['Home.Ownership'],drop_first=True)
loansData['Own_other'] = dummies['OTHER']
loansData['Own_own'] = dummies['OWN']
loansData['Own_rent'] = dummies['RENT']

loansData['Int_inc_other'] = loansData['Own_other'] * loansData['Monthly.Income']
loansData['Int_inc_own'] = loansData['Own_own'] * loansData['Monthly.Income']
loansData['Int_inc_rent'] = loansData['Own_rent'] * loansData['Monthly.Income']

# Setting the dependent variable 
y = loansData['Interest.Rate']

# Setting different matrixes for independent variables

X_1 = loansData['Monthly.Income']
X_2 = loansData[['Monthly.Income', 'Own_other', 'Own_own', 'Own_rent']]
X_3 = loansData[['Monthly.Income', 'Own_other', 'Own_own', 'Own_rent', 'Int_inc_other', 'Int_inc_own', 'Int_inc_rent']]
# fit an OLS model

X_1 = sm.add_constant(X_1)   
est_1 = sm.OLS(y,X_1).fit()


print (est_1.summary())

X_2 = sm.add_constant(X_2)   
est_2 = sm.OLS(y,X_2).fit()


print (est_2.summary())

X_3 = sm.add_constant(X_3)   
est_3 = sm.OLS(y,X_3).fit()


print (est_3.summary())