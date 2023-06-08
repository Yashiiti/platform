

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import sys
apl = yf.Ticker('GOOG')
# apl

# data = apl.history()
# data.head()
# #we will be taking the data from 2010, but here it starts from 2023 April , ie, 1 month data.

# data.tail()

# """**Taking start date and end date for the company's stock details**

# """

start_time = pd.to_datetime('2010-01-01')
end_time = pd.to_datetime('2023-05-18')
stock = ['MSFT']
data = yf.download(stock, start = start_time, end = end_time)

# """###:The accessed Data of APPLE from (January 1, 2010) to Current Date (May 18, 2023)"""

# print(data)
# print(type(data))


data.reset_index(inplace=True)
# data

data.to_csv("MSFT.csv")


"""###Visualizing the Data : (Stock Price --> "Close") """
# data= pd.read_csv("AAPL.csv")
# # print(data)
# p=["x=op","y=cl","x-y"]
# stk_close = data.reset_index()['Close']
# stk_open = data.reset_index()['Open']
# close=[]
# open=[]
# for i in range(len(stk_close)):
#     close.append(stk_close[i])
#     open.append(stk_open[i])
# alpha=[]
# # t = sys.stdin.readlines()
# # t=input("enter expression")
# s=0
# for i in range(len(close)-1):
#     cl=close[i]
#     op=open[i]
#     t=p[-1]
#     for pl in p:
#         eval(pl)
#     z=eval(t)
#     if z>0:
#         s+=close[i+1]-close[i]
#         alpha.append(s)
#     else:
#         s+=close[i]-close[i+1]
#         alpha.append(s)
#     # alpha.append(z)
#     print(alpha[i])
# plt.plot(alpha)
# plt.show()

# print(stk_close)
# stk_close

# Dates = data.reset_index()['Date']
# Dates
# print(Dates)

# plt.plot(Dates, stk_close, color = 'red')
# plt.plot(Dates, stk_open, color = 'green')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.title('Stock Price of Apple Inc. (AAPL)')
# plt.show()
# plt.savefig('apple.png')




