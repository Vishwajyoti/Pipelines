"""
Created on Wed Aug 28 17:32:38 2019

@author: vispande2
"""
import argparse
import os
import pandas as pd
import gcsfs
from sklearn.model_selection import train_test_split
import datetime
import logging


def dummy_var(df):
    df_new=pd.get_dummies(df,'Catg')
    return (df_new)

def read_file(path,filename,t_size):
    file_path=os.path.join(path,filename)
    t_size=t_size
    df=pd.read_csv(file_path)
    df_dum=dummy_var(df)
    train,test=train_test_split(df_dum,test_size=t_size,random_state=10)
    return (train,test)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  
  parser = argparse.ArgumentParser(description='Feature Engineering Of the Raw Data')
  
  parser.add_argument('--path', type=str, help='Local or GCS path to the raw file.')
  parser.add_argument('--filename', type=str, help='Filename to be processed.')
  parser.add_argument('--t_size', type=float, default=0.2, help='Test size in percent')

  args = parser.parse_args()

  train,test=read_file(args.path,args.filename,args.t_size)
  output_path=args.path+'outputs/'
  tr_file=output_path+'train.csv'
  te_file=output_path+'test.csv'
  train.to_csv(tr_file,index=False)
  test.to_csv(te_file,index=False)

