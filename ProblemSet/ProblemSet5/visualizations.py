# Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Import the dataset including the dependent variable

h1b = pd.read_excel("https://www.foreignlaborcert.doleta.gov/pdf/Per\
                    formanceData/2017/H-1B_Disclosure_Data_FY17.xlsx")

# Slice data by the date that cases were submitted

isSub2016 = (h1b['CASE_SUBMITTED'] > "2015-12-31") &
             (h1b['CASE_SUBMITTED'] < "2017-1-1")
h1b = h1b[isSub2016]

# Summary case number by groupbying state

h1b_gb = h1b['CASE_NUMBER'].groupby(h1b['WORKSITE_STATE']).count()
h1b_gb = pd.DataFrame(h1b_gb)
h1b_gb = h1b_gb.reset_index()

# Replace state abbreviation by state name

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

h1b_gb['STATE'] = h1b_gb['WORKSITE_STATE'].map(states)

# Import the dataset including the first independent variable

gdp = pd.read_csv("SAGDP2N__ALL_AREAS_1997_2018.csv")

# Slice data to select rows with summary data

isAllInd = gdp['Description'] == 'All industry total'
gdp = gdp[isAllInd]

# Build a new variable: GDP growth rate of 2016
# The purpose is to make it as a new independent variable

gdp = gdp.reset_index(drop=True)
gdp['2015'] = gdp['2015'].astype(float)
gdp['2016'] = gdp['2016'].astype(float)
gdp['2016GrowRate'] = (gdp['2016']-gdp['2015'])/gdp['2015']
gdp = pd.DataFrame(gdp)
gdp = gdp.loc[1:,:]

# Import the dataset including the third independent variable

firmN = pd.read_excel("https://www2.census.gov/programs-surveys/susb/tables/20\
                      16/state_naicssector_2016.xlsx?#",skiprows=6)

# Slice data to get rid of the first two NULL rows

firmN = firmN.loc[2:,:]

# Summary firm number by groupbying state

firmN_gb = firmN['NUMBER OF FIRMS'].groupby(firmN['STATE DESCRIPTION']).sum()
firmN_gb = pd.DataFrame(firmN_gb)
firmN_gb = firmN_gb.reset_index()

# Change variable names for merge easily, and merge three DataFrame

h1b_gb.columns = ['WorkState','CaseNumber','GeoName']
firmN_gb.columns = ['GeoName','firmNumber']

result = pd.merge(firmN_gb,gdp,on='GeoName')
result = pd.merge(result,h1b_gb,on='GeoName')

# Save data locally

result.to_excel(r'PS5result.xlsx')

# The first visual: the distribution of case number
# If there is no enough variations, I would change dependent variable

plt.style.use('ggplot')
sns.distplot(result['CaseNumber'],kde=True,rug=False,bins=30)
plt.title('Distribution of H1b case number')
plt.savefig("figure1.png",dpi=100)

# The plot shows enough variations of case number in different states

# The second visual: the scatter plot between case number and gdp
# A fit live will be added to the scatter plot

plt.scatter(result['CaseNumber'],result['firmNumber'],alpha = 0.15,marker='o')
plt.plot(np.unique(result['CaseNumber']),
         np.poly1d(np.polyfit(result['CaseNumber'],
                              result['firmNumber'],
                               1))(np.unique(result['CaseNumber'])),
         color='green', linestyle="--", linewidth=2)
plt.ylabel('Number of Firms')
plt.xlabel('Number of Cases')
plt.title('Relationship between number of firms and H1b Cases')
plt.savefig('figure2.png',dpi=100)

# The scatter shows strong positive linear relationship

# The third visual: the seaborn scatter and fitted line plot

sns.jointplot(x='2016GrowRate',y='CaseNumber',data=result,kind="reg")
plt.ylabel('H1b Case Number')
plt.xlabel('GDP Growth Rate for 2016')

plt.savefig('figure3.png',bbox_inches = 'tight')
