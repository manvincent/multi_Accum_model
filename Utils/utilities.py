from __future__ import division
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def is_odd(num):
   return num % 2 != 0

def cis_trans(data):
    cisData = data.loc[is_odd(data.Data_Split)]
    transData = data.loc[~is_odd(data.Data_Split)]
    return(cisData, transData)