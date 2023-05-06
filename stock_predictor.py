import yfinance as yf
import pylab
import numpy
import random
import pandas_market_calendars as mcal
import pandas as pd
import datetime

random.seed(0)

ref_date = numpy.datetime64('1970-01-01')  # set reference date
stock = input("Enter the stock: ")
start = input("Enter starting date in yyyy-mm-dd format ")
end = input("Enter ending date in yyyy-mm-dd format ")

data = yf.download(stock, start, end)
closing_prices = pylab.array(data['Close'].to_list())
times = pylab.array(data.index)
nyse = mcal.get_calendar('NYSE')

# Convert end date to pandas Timestamp object
end = pd.Timestamp(end)

# Get schedule for NYSE calendar
schedule = nyse.schedule(start_date = end, end_date = end + datetime.timedelta(days = 7))

# Get next trading day after end
next_trading_day = schedule.index[1].strftime('%Y-%m-%d')
next_trading_day = (pd.Timestamp(next_trading_day) - ref_date) / numpy.timedelta64(1, 'D')

# Plotting the data
pylab.plot(times, closing_prices)
pylab.xlabel('Date')
pylab.ylabel('Stock Price')

def rSquared(observed, predicted):
  error = ((predicted - observed) ** 2).sum()
  meanError = error/len(observed)

  return 1 - (meanError / numpy.var(observed))

def splitData(xVals, yVals):
  toTrain = random.sample(range(len(xVals)),len(xVals) // 2)

  trainX, trainY, testX, testY = [], [], [], []
  for i in range(len(xVals)):
    if i in toTrain:
      trainX.append(xVals[i])
      trainY.append(yVals[i])
    else:
      testX.append(xVals[i])
      testY.append(yVals[i])
  return trainX, trainY, testX, testY

numTrials = 100
degrees = (1, 2, 3, 4)
r2_dict = {}
predictions = {}
for d in degrees:
  r2_dict[d] = []
  predictions[d] = []

for i in range(numTrials):
  trainX, trainY, testX, testY = splitData(times, closing_prices)

  trainX = (trainX - ref_date).astype('timedelta64[D]').astype('float64')  # convert to float64
  testX = (testX - ref_date).astype('timedelta64[D]').astype('float64')

  for d in degrees:
    model = pylab.polyfit(trainX, trainY, d)
    estYVals = pylab.polyval(model, testX)
    r2 = rSquared(estYVals, testY)
    prediction = pylab.polyval(model, next_trading_day)
    r2_dict[d].append(r2)
    predictions[d].append(prediction)

for d in degrees:
  mean_r2 = round(sum(r2_dict[d]) / len(r2_dict[d]), 2)
  sd_r2 = round(numpy.std(r2_dict[d]), 2)
  mean_prediction = round(sum(predictions[d]) / len(predictions[d]), 2)
  sd_prediction = round(numpy.std(predictions[d]), 2)
  if mean_r2 < 0.6:
    print("The regression model doesn't fit the data well, for degree", d)
  else:  
    print("For degree", d, "regression model, mean R-square value is", mean_r2, "with std =", sd_r2)
    print("The mean predicted value is", mean_prediction, "with std =", sd_prediction)
    print("According to degree", d, "model there is a 95% probability that the predicted value is in the range", round(mean_prediction - 1.96 * sd_prediction, 2), "-",
         round(mean_prediction + 1.96 * sd_prediction, 2))
    
  print("\n")

pylab.show()
