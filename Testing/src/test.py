"""
Created on Mon Aug 26 19:43:52 2019

@author: vispande2
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging
from sklearn.metrics import r2_score,mean_squared_error
import argparse
import gcsfs

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Testing the Model on test data.')
    parser.add_argument('--path', type=str, help='Local or GCS path to the test file')
    parser.add_argument('--target', type=str, help='Dependent varaible name.')
    
    args = parser.parse_args()
    
    test=pd.read_csv(args.path+'outputs/test.csv')
    X=test.drop(args.target,axis=1)
    Y=test[[args.target]]
    Y.columns=['Actual']
    fs = gcsfs.GCSFileSystem(token='cloud')
    model=pickle.load(fs.open((args.path+'models/model.pkl'),'rb'))
    Predict=model.predict(X)
    Y['Predict']=Predict
    Y.to_csv(args.path+'test_pred/ActualPredict.csv')
    print ("Test rmse is:",np.round(np.sqrt(mean_squared_error(Y['Actual'],Y['Predict'])),3))
    print ("Test RSquared is:",np.round(np.sqrt(r2_score(Y['Actual'],Y['Predict'])),3))
