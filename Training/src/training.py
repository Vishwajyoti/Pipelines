"""
Created on Tue Aug 26 11:56:52 2019

@author: vispande2

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import logging
import argparse
import gcsfs
import json
import ast
import pickle

def grid_search(df,target,parameter):
    X_train=df.drop(target,axis=1)
    Y_train=df[[target]]
    rf_model=RandomForestRegressor(random_state=42)
    g_search=GridSearchCV(estimator = rf_model, param_grid = parameter,cv = 5, n_jobs = -1)
    g_search.fit(X_train,Y_train)
    g_best=g_search.best_estimator_
    return (g_best)

def rand_search(df,target,parameter):
    X_train=df.drop(target,axis=1)
    Y_train=df[[target]]
    rf_model=RandomForestRegressor(random_state=42)
    r_search=RandomizedSearchCV(estimator = rf_model,param_distributions = parameter,cv = 5, n_jobs = -1,n_iter=100)
    r_search.fit(X_train,Y_train)
    r_best=r_search.best_estimator_
    return (r_best)
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Traiining the Model on data')
    parser.add_argument('--path', type=str, help='Local or GCS path to the training file')
    parser.add_argument('--target', type=str, help='Dependent varaible name.')
    parser.add_argument('--h_param', type=str, help='Hyperparameter for tuning,dictonary in json format')
    parser.add_argument('--search_type', type=int, default=1, help='Type of hyper-parameter search: Grid ->1, Random->2')
    
    args = parser.parse_args()
    train=pd.read_csv(args.path+'outputs/train.csv')
    param=ast.literal_eval(args.h_param)
    if args.search_type==1:
        model=grid_search(train,args.target,param)
    else:
        model=rand_search(train,args.target,param)
    X=train.drop(args.target,axis=1)
    Y=train[[args.target]]
    model.fit(X,Y)
    fs = gcsfs.GCSFileSystem()
    pickle.dump(model,fs.open((args.path+'models/model.pkl'),'wb'))
