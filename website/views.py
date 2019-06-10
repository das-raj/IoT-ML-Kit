from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse


#ML models imports
import pandas as pd
import numpy as np

import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import datetime

# Create your views here.
def index(request):
	if request.method == "GET":
		return render(request, 'index.html')


def input_form(request):
	if request.method=="GET":
		return render(request, 'form.html')
	elif request.method=="POST":
		channel_id = request.POST.get('channel_id')
		field_no = request.POST.get('field_number')
		model = request.POST.get('model')
		print('[+]Thingspeak Credentials: ', channel_id,field_no,model)

		url_thingspeak = 'https://thingspeak.com/channels/'+channel_id+'/fields/'+field_no+'.csv'
		request.session['website'] = 'IOT_ML_KIT'
		#print('[+]SessionKey: '+request.session.session_key)

		#filename = model+"_"+str(request.session.session_key)+'.png'
		filename = model+"_"+str(datetime.datetime.now().timestamp())+'.png'
		if model=="LR":
			linearmodel(url_thingspeak, 'static/images/'+filename)
		elif model=="SV":
			supportvector(url_thingspeak, 'static/images/'+filename)
		elif model=="RF":
			randomforest(url_thingspeak, 'static/images/'+filename)

		return render(request, 'data_visualize.html', {'filename':filename})

def data_preprocess(url):
	data = pd.read_csv(url)

	# Preprocessing string to timestamp
	#2019-06-03 10:47:42 UTC
	for i in range(len(data['created_at'])):
	    dt = datetime.datetime.strptime(data['created_at'][i], '%Y-%m-%d %H:%M:%S UTC')
	    data['created_at'][i] = dt.timestamp()

	#  (X, Y) - (Timestamp, Temperature)
	X = data['created_at'].to_numpy().reshape(-1,1)
	y = data['field1'].to_numpy().reshape(-1,1)

	return X,y

def supportvector(url, filename):
	X,y = data_preprocess(url)

	# Fitting SVR to the dataset
	from sklearn.svm import SVR
	regressor_svr = SVR(kernel = 'rbf')
	regressor_svr.fit(X, y)

	# Visualising the SVR results (for higher resolution and smoother curve)
	import matplotlib.pyplot as pltsv
	pltsv.scatter(X, y, color = 'red')
	pltsv.plot(X, regressor_svr.predict(X), color = 'blue')
	pltsv.title('Support Vector')
	pltsv.xlabel('Timestamp')
	pltsv.ylabel('Temperature')
	#plt.show()
	pltsv.savefig(filename)
	print('[+]Image Saved')
	

def randomforest(url, filename):
	X,y = data_preprocess(url)

	# Fitting Random Forest Regression to the dataset
	regressor_rf = RandomForestRegressor(n_estimators = 10000, random_state = 0)
	regressor_rf.fit(X, y)

	# Visualising the SVR results (for higher resolution and smoother curve)
	import matplotlib.pyplot as plt
	plt.scatter(X, y, color = 'red')
	plt.plot(X, regressor_rf.predict(X), color = 'blue')
	plt.title('Random Forest')
	plt.xlabel('Timestamp')
	plt.ylabel('Temperature')
	#plt.show()
	plt.savefig(filename)
	print('[+]Image Saved')


def linearmodel(url, filename):
	X,y = data_preprocess(url)

	# Splitting Data into Training & Testing
	# TODO Split in 80:20, i.e. 10 for test and 20 for training
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

	# WELL DONE! YOUR DATA IS PREPROCESSED

	# Fitting SIMPLE LINEAR REGRESSION to the Training Set
	from sklearn.linear_model import LinearRegression
	regressor_lr = LinearRegression()
	regressor_lr.fit(X_train, y_train)

	# Visualising the Training set results
	import matplotlib.pyplot as pltlm
	pltlm.scatter(X_train, y_train, color='red', label='predicted')
	pltlm.plot(X_train, regressor_lr.predict(X_train), color='blue', label='actual')
	pltlm.title('Linear Regression')
	pltlm.xlabel('Timestamp')
	pltlm.ylabel('Temperature')
	pltlm.legend(loc='upper left', numpoints=1)
	#plt.show()
	pltlm.savefig(filename)
	print('[+]Image Saved')


