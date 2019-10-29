# Import packages

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Import data for analyzing

df1 = pd.read_excel('PS5result.xlsx')
df1['2016GDP'] = df1['2016']

# Build a constant variable

df1['const'] = 1

# Run OLS model and print its result

reg1 = sm.OLS(endog=df1['CaseNumber'], exog=df1[['const', 'firmNumber','2016GDP',
              '2016GrowRate']], missing='drop')

results = reg1.fit()

print(results.summary())

# The overall model and each independent variable show statistical significance
# However, to make 2SLS easier, I only keep 2016GDP as the independent variable

reg1 = sm.OLS(endog=df1['CaseNumber'], exog=df1[['const','2016GDP']],
               missing='drop')
results = reg1.fit()
print(results.summary())

# Import data to add the first instrument to the model

df_i = pd.read_excel('institution.xlsx')
df_i.columns = ['GeoName','instituNumber']

# Merge the data to the main dataset

instrument1 = pd.merge(df1,df_i,on='GeoName')

# Use the two-stage least squares (2SLS) regression
# The first stage: regress the gdp on the instrument

instrument1['const']=1

result_i1 = sm.OLS(instrument1['2016GDP'],instrument1[['const',
                  'instituNumber']],missing='drop').fit()
print(result_i1.summary())

# The second stage: retrieve the predicted values of gdp
# Use the predicted gdp to run again the main model

instrument1['predicted_gdp'] = result_i1.predict()

result_i12 = sm.OLS(instrument1['CaseNumber'],instrument1[['const',
                    'predicted_gdp']]).fit()
print(result_i12.summary())

# Import data to add the second instrument to the model

df_i2 = pd.read_excel('election2016.xlsx')
df_i2 = df_i2.iloc[:,0:4]
df_i2.columns = ['GeoName','em','DemoNumber','DemoPer']

# Convert the percentage string to float

for i in range(len(df_i2)):
    df_i2.at[i,"DemoPer"] = float(df_i2.at[i,"DemoPer"].replace('%',''))

instrument2 = pd.merge(df1,df_i2,on='GeoName')
instrument2['2016GDP']=instrument2['2016']
instrument2['DemoPer'] = instrument2['DemoPer'].astype(float)

# The first stage: regress the gdp on the instrument

instrument2['const']=1

result_i2 = sm.OLS(instrument2['2016GDP'],instrument2[['const','DemoPer']],
                   missing='drop').fit()
print(result_i2.summary())

# The second stage: retrieve the predicted values of gdp
# Use the predicted gdp to run again the main model

instrument2['predicted_gdp'] = result_i2.predict()

result_i22 = sm.OLS(instrument2['CaseNumber'],instrument2[['const',
                    'predicted_gdp']]).fit()
print(result_i22.summary())
