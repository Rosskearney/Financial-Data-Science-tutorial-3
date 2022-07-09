#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:11:46 2022

@author: rosskearney
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import shutil
import os
import tensorflow as tf
import pandas_ta as ta
from sklearn.model_selection import train_test_split
import yfinance

cols = 32

def create_model():
    # Random seed for repeatability
  np.random.seed(100)
  tf.random.set_seed(100)
  return tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

df = pd.DataFrame() # Empty DataFrame
# df = df.ta.ticker("BTC-USD")

df = df.ta.ticker("EURUSD=X")

df['Return'] = np.log(df['Close']/df['Close'].shift(1))


# =============================================================================
# 
# =============================================================================

# Create rolling averages, easy to change as '30' is 30 days and '5' is 5 days
df['CURrolling30'] = df['Close'].rolling(5).mean()
df['CURrolling5'] = df['Close'].rolling(30).mean()

# drop empty values
df.dropna(inplace = True)

# Create new column, if rolling 5 is > rolling 30, plot a 'buy' signal
# if rolling 5 < rolling 30, plot 'sell' signal 
# (plot runs to high/low of highest/lowest stock price for clean graph)
df['Direction'] = np.where(df['CURrolling5'] > df['CURrolling30'], 1, 0)


# Plot the standard Adj. Close price with label
plt.plot(df['Close'], label = 'CUR Close Price')

# Plot the buy/sell signal but with no legend
plt.plot(df['buySell'], label = '_nolegend_')

# Plot both sets of rolling data
plt.plot(df['CURrolling30'], label = '30 Day Rolling Mean')
plt.plot(df['CURrolling5'], label = '5 Day Rolling Mean')

# graph lables & legend location
plt.ylabel('Close Price $')
plt.xlabel('Date')
plt.title('CUR Returns vs Rolling Averages with Buy/Sell signals')       
plt.legend(loc = 2)
plt.show()



# =============================================================================
# 
# =============================================================================


# create lagged returns
cols=[]
lags = range(1,5)
for lag in lags:
    df[f'rolling{lag}'] = df['Close'].rolling(20*lag).mean()
    col =f'rollingdir{lag}'
    df[col] = np.where(df['CURrolling{lag}'] > df['CURrolling{lag}'].shift(lag),
                       1, 0)
    cols.append(col)

df.dropna(inplace=True)

# df['direction'] = np.where(df['Return'] > 0, 1, 0)

df['Direction'] = np.where(df['BuySell'] > df['BuySell'], 1, 0)


# split the dataset in training and test datasets
train, test = train_test_split(df.dropna(), test_size=0.4, shuffle=False)


# create the model
model = create_model()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
r = model.fit(train[cols], 
              train['direction'], 
              epochs=5,
              validation_data=(test[cols], test['direction']), 
              callbacks=[tensorboard_callback]) #verbose=False
OUTPUT_DIR = "./export/savedmodel"
shutil.rmtree(OUTPUT_DIR,ignore_errors=True)
EXPORT_PATH = os.path.join(OUTPUT_DIR,datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
tf.saved_model.save(model,EXPORT_PATH)

pred = model.predict(test[cols]) 
trade = pred > 0.5 + pred.std()
profits = test.loc[trade, 'Return'] 
print('Scheme Info. Ratio:' + "{:.2f}".format(np.sqrt(365)*profits.mean()/profits.std()))
print('Underlying Info. Ratio:' + "{:.2f}".format(np.sqrt(365)*test['Return'].mean()/test['Return'].std()))




