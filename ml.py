import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# Process data to establish correlation values
def process_data_for_labels(ticker):
    days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    
    for i in range(1, days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        
    df.fillna(0, inplace=True)
    return tickers, df, days

       
# Helper function for buy/sell/hold labels 
def buy_sell_hold(*args):
    cols = [c for c in args]
    req = 0.025
    for col in cols:
        if col > req:
            return 1
        elif col < -req:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df, days = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                              *[df['{}_{}d'.format(ticker, i)]for i in range(1, days+1)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data Spread:', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), 
                            ('knn', neighbors.KNeighborsClassifier()), 
                            ('rfor', RandomForestClassifier())])
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    
    predictions = clf.predict(X_test)
    print('Predicted Spread:', Counter(predictions))
    
    return confidence

do_ml('BAC')