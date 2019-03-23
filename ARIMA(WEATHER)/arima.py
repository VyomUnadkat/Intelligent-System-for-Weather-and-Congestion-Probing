
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('weather.csv')
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(12,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
print("for next week")


predictions_week = list()
test_week = predictions[-7:]

for t in range(len(test_week)):
	model = ARIMA(history, order=(12,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions_week.append(yhat)
	obs = test_week[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))




error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot testing
pyplot.plot(test, color='blue', label='actual temperature')
pyplot.plot(predictions, color='red', label='predicted temperature')
pyplot.show()

# plot predictions
pyplot.plot(test_week, color='blue', label='actual temperature')
pyplot.plot(predictions_week, color='red', label='predicted temperature')
pyplot.show()


