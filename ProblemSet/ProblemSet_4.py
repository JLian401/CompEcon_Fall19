#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from geopy.distance import vincenty


# Read original dataset

df = pd.read_excel(
    "C:/Users/Jie Lian/CompEcon_Fall19/Matching/radio_merger_data.xlsx")


# Build data array function for the input of object functions

def arr_dat(df):
    '''
    Build data array function as the input of the obejct functions

    Args:
        df: the original "observed" matches dataframe

    Returns:
        da: the "counterfactual" matches dataframe
    '''

    # Subset the data into buyer related dataframe and target related dataframe

    df_buyer = df[["year","buyer_id","buyer_lat","buyer_long",
                   "num_stations_buyer","corp_owner_buyer"]]
    df_target = df[["target_id","target_lat","target_long","price",
                    "hhi_target","population_target"]]

    # Transform the Pandas DataFrame to Numpy Array and leverage its broadcasting

    arr_buyer = pd.DataFrame(df_buyer).to_numpy()
    arr_target = pd.DataFrame(df_target).to_numpy()

    arr_zero = np.zeros((len(arr_target),np.shape(arr_buyer)[1]),dtype=np.float)
    arr_data = np.array(["year","buyer_id","buyer_lat","buyer_long",
                         "num_stations_buyer","corp_owner_buyer","target_id",
                         "target_lat","target_long","price","hhi_target",
                         "population_target"])

    for i in range(len(arr_buyer)):
        arr_mid_buyer = arr_zero + arr_buyer[i]
        arr_hstack = np.hstack((arr_mid_buyer,arr_target))
        arr_data = np.vstack((arr_data,arr_hstack))

    # Add columns name, drop duplicates, and give up "observed" rows

    df_data = pd.DataFrame(arr_data,columns=arr_data[0,:])
    df_data = df_data.drop(df_data.index[0])
    df_data = df_data.drop_duplicates()
    df_data = df_data[df_data['buyer_id'] != df_data['target_id']]
    df_data = df_data.reset_index()
    df_data = pd.DataFrame(df_data,dtype=float)

    return df_data


# Contruct distance variable by calling vincenty class of geopy.distance package

def distance(df):
    for i in range(len(df["buyer_id"])):
        df.at[i,"distance"] = vincenty(
            (df.at[i,"buyer_lat"],df.at[i,"buyer_long"]),
            (df.at[i,"target_lat"],df.at[i,"target_long"])).miles
    return df


# Seperate input data by markets,2007 and 2008

df_o_2007 = df[df["year"]==2007]
df_o_2008 = df[df["year"]==2008]

# Call arr_dat() and distance() to get objective function input ready

df_f_2007 = arr_dat(df_o_2007)
df_f_2007 = distance(df_f_2007)
df_f_2008 = arr_dat(df_o_2008)
df_f_2008 = distance(df_f_2008)

# Reset index for all four dataframes
df_o_2007["ind"] = df_o_2007["buyer_id"]
df_o_2008["ind"] = df_o_2008["buyer_id"]
df_o_2007 = df_o_2007.set_index(["ind"])
df_o_2008 = df_o_2008.set_index(["ind"])

df_f_2007["buyer_id"] = pd.to_numeric(df_f_2007["buyer_id"],
                                      errors='coerce').astype(np.int64)
df_f_2007["target_id"] = pd.to_numeric(df_f_2007["target_id"],
                                       errors='coerce').astype(np.int64)
df_f_2008["buyer_id"] = pd.to_numeric(df_f_2008["buyer_id"],
                                      errors='coerce').astype(np.int64)
df_f_2008["target_id"] = pd.to_numeric(df_f_2008["target_id"],
                                       errors='coerce').astype(np.int64)


df_o_2007.head(5)


# Objective function

def obj_func_1(Theta):
    """
    The object function for the model 1.

    Args:
        Theta: A length 2 tuple, model parameters (alpha, beta)

    Returns:
        score function with alpha and beta as variables
    """

    alpha, beta = Theta

    def payoff_1(index,data):
        payoff = data.at[index,"num_stations_buyer"] *
        data.at[index,"population_target"] + alpha *
        data.at[index,"corp_owner_buyer"] *
        data.at[index,"population_target"] + beta*data.at[index,"distance"]
        return payoff

    # calculate payoff in different market, 2007 and 2008
    score_2007 = 0
    for i in range(1,len(df_o_2007["buyer_id"])+1):
        for j in range(1,len(df_o_2007["buyer_id"])+1):
            ind_o_i = i
            ind_o_j = j
            ind_f_i = df_f_2007.loc[(df_f_2007["buyer_id"]==j) &
                                          (df_f_2007["target_id"]==i)].index[0]
            ind_f_j = df_f_2007.loc[df_f_2007["buyer_id"]==i &
                                         df_f_2007["target_id"]==j].index[0]
            if i == j:
                pass
            else:
                if (payoff_1(ind_o_i,df_o_2007) + payoff_1(ind_o_j,df_o_2007)) >
                 (payoff_1(ind_f_i,df_f_2007) + payoff_1(ind_f_j,df_f_2007)):
                    score_2007 += 1
                else:
                    pass

    score_2008 = 0
    for i in range(1,len(df_o_2008["buyer_id"])+1):
        for j in range(1,len(df_o_2008["buyer_id"])+1):
            ind_o_i = i
            ind_o_j = j
            ind_f_i = df_f_2008.loc[df_f_2008["buyer_id"]==j &
                                          df_f_2008["target_id"]==i].index[0]
            ind_f_j = df_f_2008.loc[df_f_2008["buyer_id"]==i &
                                         df_f_2008["target_id"]==j].index[0]

            if i == j:
                pass
            else:
                if (payoff_1(ind_o_i,df_o_2008) + payoff_1(ind_o_j,df_o_2008)) >
                (payoff_1(ind_f_i,df_f_2008) + payoff_1(ind_f_j,df_f_2008)):
                    score_2008 += 1
                else:
                    pass

    inv_total = -(score_2007 + score_2008)
    return inv_total


# Objective function

def obj_func_2(Sigma):
    """
    The object function for the model 2.

    Args:
        Theta: A length 4 tuple, model parameters (delta,alpha,gamma,beta)

    Returns:
        score function with delta,alpha,gamma,beta as variables
    """

    delta,alpha,gamma,beta = Sigma

    def payoff_2(index,data):
        payoff = delta * data.at[index,"num_stations_buyer"] *
        data.at[index,"population_target"] + alpha *
        data.at[index,"corp_owner_buyer"] * data.at[index,"population_target"] +
        beta*data.at[index,"distance"] + gamma * data.at[index,"hhi_target"]
        return payoff

    # calculate payoff in different market, 2007 and 2008
    score_2007 = 0
    for i in range(1,len(df_o_2007["buyer_id"])+1):
        for j in range(1,len(df_o_2007["buyer_id"])+1):
            ind_o_i = df_o_2007.index[df_o_2007["buyer_id"]==i].tolist()[0]
            ind_o_j = df_o_2007.index[df_o_2007["buyer_id"]==j].tolist()[0]
            ind_f_i = df_f_2007.loc[(df_f_2007["buyer_id"]==j) &
                                          (df_f_2007["target_id"]==i)].index[0]
            ind_f_j = df_f_2007.loc[df_f_2007["buyer_id"]==i &
                                         df_f_2007["target_id"]==j].index[0]
            if i == j:
                pass
            else:
                if (payoff_2(ind_o_i,df_o_2007) + payoff_2(ind_o_j,df_o_2007)) >
                (payoff_2(ind_f_i,df_f_2007) + payoff_2(ind_f_j,df_f_2007)):
                    score_2007 += 1
                else:
                    pass

    score_2008 = 0
    for i in range(1,len(df_o_2008["buyer_id"])+1):
        for j in range(1,len(df_o_2008["buyer_id"])+1):
            ind_o_i = df_o_2008.index[df_o_2008["buyer_id"]==i].tolist()[0]
            ind_o_j = df_o_2008.index[df_o_2008["buyer_id"]==j].tolist()[0]
            ind_f_i = df_f_2008.loc[df_f_2008["buyer_id"]==j &
                                          df_f_2008["target_id"]==i].index[0]
            ind_f_j = df_f_2008.loc[df_f_2008["buyer_id"]==i &
                                         df_f_2008["target_id"]==j].index[0]

            if i == j:
                pass
            else:
                if (payoff_2(ind_o_i,df_o_2008) + payoff_2(ind_o_j,df_o_2008)) >
                (payoff_2(ind_f_i,df_f_2008) + payoff_2(ind_f_j,df_f_2008)):
                    score_2008 += 1
                else:
                    pass

    inv_total = -(score_2007 + score_2008)
    return inv_total


# Use Nelder-Meade method to calculate maximum. Failed. 

import scipy.optimize as opt
from scipy.optimize import differential_evolution

Theta0 = (10,30)

bnds = [(-1000,1000),(-1000,1000)]

result = opt.minimize(obj_func_1, Theta0, bounds=bnds,
                      method='Nelder-Mead',tol=1e-15)
