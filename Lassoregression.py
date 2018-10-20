"""
==============================
Lasso Regression Model
==============================

This program allow the user to train a LASSO regression model on training data
and identify features that are most relevant to the response vector.
As input, the code accept a feature matrix (samples of x features)
as well as a response vector (correspond to different samples).

Output generated contains coeffiecient correspoding to the list of features in the LASSO mode.

"""
print(__doc__)


#Importing libraries.
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


#Load the dataset
data = pd.read_csv("Sample_features.csv")

#Function for LASSO regression and to generate LASSO co-effiencient and intercept
def lasso_regression(data, predictors, alpha):
    #Fit the Lasso model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
        
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

#Initialize predictors to all features
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1e-1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(2,17)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data,predictors, alpha_lasso[i])

#Saving the generated coefficients of x features    
coef_matrix_lasso.to_csv('Output_Coef_Xi.csv')

print(coef_matrix_lasso)

