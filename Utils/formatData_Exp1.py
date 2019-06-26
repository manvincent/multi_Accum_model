import numpy as np
import pandas as pd

def formatSimData(df):
    # Rename stim and response (code) columns
    df.rename(columns={'stim':'stim_Code', 'Response':'Response_Code'},inplace=True)
    # Replace stim and response codes with labels
    df['Response_Type'] = df['Response_Code'].replace([0,1,2],['Gain  ','Cntrl ','NoLoss'])
    df['stim'] = df['stim_Code'].replace([0,1,2],['Gain  ','Cntrl ','NoLoss'])
    # Specify the outcome magnitude for each trial
    df['Money_Amount'] = df['gainVal']
    # Convert RT to sec
    df['rt_sec'] = np.divide(df['rt'],1000,dtype=float)
    # Calculate accuracy columns
    df['Accuracy'] = np.where(df['Response_Code']==df['stim_Code'], 1, 0)
    df['Accuracy_inTime'] = np.where((df['Accuracy']==1) & (df['rt']<400), 1, 0)
    
    return df
    
def formatExpData(df):
    # Specify the outcome magnitude for each trial
    df['Money_Amount'] = df['gainVal']
    # Convert RT to sec
    df['rt_sec'] = np.divide(df['rt'],1000,dtype=float)  
    return df