# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:52:55 2018

@author: MA5042387
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.models import model_from_json
import matplotlib.dates as mdates
from sklearn.externals import joblib
from collections import OrderedDict
import json
import os
import re
import ast
import warnings

warnings.filterwarnings("ignore")

inputpath = 'input\\'
indexpath = 'index\\'
modelpath = 'model\\'
imagepath = 'images\\'
anomaly = 'anomaly\\'
submodelpath = modelpath + 'submodels\\'

months = mdates.MonthLocator(interval=6)
monthsFmt = mdates.DateFormatter('%Y-%m')
freq_dict = {2:'W', 3:'M', 4:'Y'}
freq_to_days = {1:1,2:52, 3:12, 4:1}
pvalues = [0,1,2,3,4,5]
qvalues = [0,1,2,3]
freq_degree = {1:12,2:12,3:4,4:1}
str_to_freq = {'daily':1,'weekly':2,'monthly':3,'yearly':4}

app_root=os.path.dirname(os.path.abspath(__file__))+"\\"


def validate(file,date,tgt):
    newfile = file.set_index(date)
    newfile.index = pd.to_datetime(newfile.index)
    #print(newfile.info())
    newfile['year'] = newfile.index.year
    newfile['month'] = newfile.index.month
    newfile['week'] = newfile.index.weekofyear
    newfile['date'] = newfile.index.date
    #print(newfile.head(15))
    agg1 = newfile.groupby(by='year').count()[tgt]
    agg2 = newfile.groupby(by=['year','week']).count()[tgt]
    agg3 = newfile.groupby(by=['year','month']).count()[tgt]
    agg4 = newfile.groupby(by=['date']).count()[tgt]
    #print(agg2)
    #print(newfile[(newfile['year']==2013) ])
    print(agg4.sum()/(len(agg4)*24))
    if agg1.sum()/(len(agg1)*365) >= 0.85: return 1
    elif (agg4.sum()/(len(agg4)*24))>=0.9: return 1
    elif agg2.sum()/(len(agg1)*52) >=0.9: return 2
    elif agg3.sum()/(len(agg1)*12) >=0.9: return 3
    
    elif agg1.sum()/(len(agg1))==1: return 4
    else: return 5

def merge_data(df, holiday_df, datefield, tgtfield, other_regressors,train_dict):
    print(other_regressors)
    train_dict['holidayfile']=0;train_dict['other_regressors']=0
    train_dict['regressor_fields']=[]; train_dict['holidayfields']=[]
    if holiday_df is not None and other_regressors is not None:
        train_dict['holidayfile']=1;train_dict['other_regressors']=1
        train_dict['regressor_fields']=other_regressors
        if tgtfield is not None: newdf = df[[datefield,tgtfield]+other_regressors]
        else: newdf = df[[datefield]+other_regressors]
        newdf[datefield] = pd.to_datetime(newdf[datefield])
        holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
        holidays = holiday_df['holiday'].unique().tolist()
        train_dict['holidayfields']=holidays
        for holiday in holidays:
            temp_df = holiday_df[holiday_df['holiday']==holiday]
            temp_df[holiday]=1
            temp_df = temp_df[['ds',holiday]]
            newdf = newdf.merge(temp_df,left_on=[datefield],right_on=['ds'], how='left')
        if tgtfield is not None: newdf = newdf[[datefield]+[tgtfield]+other_regressors+holidays]
        else: newdf = newdf[[datefield]+other_regressors+holidays]
        newdf.fillna(0,inplace=True)
        newdf = newdf.set_index(datefield)
        
    elif other_regressors is not None:
        train_dict['other_regressors']=1
        train_dict['regressor_fields']=other_regressors
        if tgtfield is not None: newdf = df[[datefield,tgtfield]+other_regressors]
        else: newdf = df[[datefield]+other_regressors]
        newdf[datefield] = pd.to_datetime(newdf[datefield])
        newdf.fillna(0,inplace=True)
        newdf = newdf.set_index(datefield)
    
    elif holiday_df is not None:
        train_dict['holidayfile']=1
        if tgtfield is not None: newdf = df[[datefield,tgtfield]]
        else: newdf = df[[datefield]]
        newdf[datefield] = pd.to_datetime(newdf[datefield])
        holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
        holidays = holiday_df['holiday'].unique().tolist()
        train_dict['holidayfields']=holidays
        for holiday in holidays:
            temp_df = holiday_df[holiday_df['holiday']==holiday]
            temp_df[holiday]=1
            temp_df = temp_df[['ds',holiday]]
            newdf = newdf.merge(temp_df,left_on=[datefield],right_on=['ds'], how='left')
        if tgtfield is not None: newdf = newdf[[datefield]+[tgtfield]+holidays]
        else: newdf = newdf[[datefield]+holidays]
        newdf.fillna(0,inplace=True)
        newdf = newdf.set_index(datefield)
    
    else:
        if tgtfield is not None: newdf = df[[datefield,tgtfield]]
        else: newdf = df[[datefield]]
        newdf[datefield] = pd.to_datetime(newdf[datefield])
        newdf.fillna(0,inplace=True)
        newdf = newdf.set_index(datefield)
    return newdf, train_dict
                        
def create_plots(series,xlabel,ylabel):
    plots ={}
    fig = plt.figure(figsize=(20,7))
    plt.plot(series,color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(xlabel + ' vs ' + ylabel)
    fig.savefig(app_root+imagepath+'tsplot.jpg')
    plt.close(fig)    
    result = seasonal_decompose(series, model='additive')
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(20,10))
    result.trend.plot(ax=axes[0], color='blue', legend=False)
    axes[0].set_ylabel('Trend')
    axes[0].set_title('Trend', fontsize=18)
    result.seasonal.plot(ax=axes[1], color='blue', legend=False)
    axes[1].set_ylabel('Seasonal')
    axes[1].set_title('Seasonal', fontsize=18)
    result.resid.plot(ax=axes[2], color='blue', legend=False)
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Residual', fontsize=18)
    fig.suptitle('Timeseries Decompose', fontsize=20)
    fig.savefig(app_root+imagepath + 'decompose_plot.jpg')
    plt.close(fig)
    plots={'plot1':app_root+imagepath+'tsplot.jpg', 'plot2':app_root+imagepath+'decompose_plot.jpg'}
    return plots

def summarize(filename,ts_col,val_col,infreq):
    infile = pd.read_csv(filename)
    dateval = ts_col
    tgtval = val_col
    freqval = str_to_freq[infreq.lower()]
    filteredfile = infile[[dateval,tgtval]]
    rtval = validate(filteredfile,dateval,tgtval)
    if rtval ==1: print('Daily Data')
    elif rtval ==2: print("Weekly Data")
    elif rtval ==3: print("Monthly Data")
    elif rtval==4: print("Yearly Data")
    else: print("Need more data")

    ifile = filteredfile.set_index(dateval)
    ifile.index = pd.to_datetime(ifile.index)
    reagg = ifile
    if int(freqval)>=rtval: 
        if int(freqval)>rtval: reagg = reagg.resample(freq_dict[int(freqval)]).mean()
        rt_plots = create_plots(reagg,dateval,tgtval)
    else: print("Frequency doesnt match")
    return rt_plots

def fit_linear(series):
    X = [i for i in range(1,len(series)+1)]
    X = np.reshape(X,(len(X),1))
    y= series
    model = LinearRegression()
    model.fit(X,y)
    pred = model.predict(X)
    return model,pred

def fit_poly(series,ts_freq):
    X = [i%freq_to_days[int(ts_freq)] for i in range(0,len(series))]
    y = series
    degree = freq_degree[int(ts_freq)]
    coef = np.polyfit(X, y, degree)
    val = coef[-1][0]
    curve = []
    for i in range(len(X)):
        value = val
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value[0])
    curve = np.reshape(curve,(len(curve),1))
    return degree, coef, curve

def predict_linear(model,X):
    y = model.predict(X)
    return np.squeeze(y)

def predict_poly(deg, coefs, X):
    value = coefs[-1][0]
    for d in range(deg):
        value += X**(deg-d) * coefs[d]
    return np.squeeze(value)
    
def remove_fit(series,fitvalues):
    values = series
    detrend_values = values-fitvalues
    return detrend_values

def dickyfuller_test(series):
    dftest = adfuller(series)
    return dftest

def acf_pacf(series):
    p1 = plot_acf(series)
    p2 = plot_pacf(series)
    return p1,p2


def evaluate_model(X_train,X_test,aorder):
    history = [x for x in X_train];pred=[]
    accuracy=0
    for i in range(len(X_test)):
        model = ARIMA(history,order=aorder)
        model_fit = model.fit(disp=0)
        forecast = model_fit.forecast()[0]
        pred.append(forecast)
        history.append(X_test[i])
    error = mean_squared_error(X_test, pred)
    accuracy = r2_score(X_test,pred)
    return error,accuracy

def get_configuration(train,test,pvals,qvals):
    score =float('inf');conf=None; acc_score=0
    for p in pvals:
        for q in qvals:
            order =(p,0,q)
            try:
                mse,acc = evaluate_model(train,test,order)
                if mse<score: 
                    score, conf=mse,order
                    acc_score = acc
            except:
                continue
    return score,acc_score,conf


def arima_predict(X_train,X_test,aorder, trend_model, seasonal_deg, seasonal_coef, in_freq):
    history = [x for x in X_train]
    pred = []
    for i in range(1,len(X_test)+1):
        model = ARIMA(history,order=aorder)
        model_fit = model.fit(disp=0)
        forecast = model_fit.forecast()[0]
        polyterm = len(history)%freq_to_days[int(in_freq)]
        forecast = forecast + predict_poly(seasonal_deg,seasonal_coef,polyterm)
        forecast = forecast + predict_linear(trend_model,len(history)+1)
        #print(forecast)
        forecast = np.exp(forecast[0])-1
        pred.append(forecast)
        history.append(X_test[i-1])
    return pred

def test_arima(modelname,testfile,datecol):
    testdf = pd.read_csv(testfile)
    jsmodel = json.load(open(app_root+modelpath+modelname+'.json'))
    filename = jsmodel['deseasonal_file']
    X_train = pd.read_csv(app_root+indexpath+filename,header=None)
    seasonal_deg = jsmodel['poly_deg']
    seasonal_coef = np.array(jsmodel['poly_coef'])
    in_freq = jsmodel['infreq']
    aorder = jsmodel['arima_order']
    trend_modelname = jsmodel['linear'] 
    testdf['temp']=0
    ifile = testdf.set_index(datecol)
    ifile.index = pd.to_datetime(ifile.index)
    ifile = ifile.resample(freq_dict[int(in_freq)]).mean()
    ts_len = len(ifile)
    trend_model = joblib.load(app_root+submodelpath+trend_modelname)
    history = [x for x in X_train.values]
    pred = {}
    for i in range(1,ts_len+1):
        try:
            model = ARIMA(history,order=aorder)
            model_fit = model.fit(disp=0)
            forecast = model_fit.forecast()[0]
            history.append(forecast)
            polyterm = len(history)%freq_to_days[int(in_freq)]
            forecast = forecast + predict_poly(seasonal_deg,seasonal_coef,polyterm)
            forecast = forecast + predict_linear(trend_model,len(history)+1)
            forecast = np.exp(forecast[0])-1
            pred[str(ifile.index[i-1].date())]=forecast
            #pred.append(forecast)
        except:
            return pred
    return pred

def train_arima(modelname,filename,ts_col,val_col,ts_freq):
    plots={}
    train_info={}
    infile = pd.read_csv(filename)
    filename = re.search(r'\\([^\\]+\.csv)',filename).group(0)
    filename = filename.replace('\\','')
    dateval = ts_col
    tgtval = val_col
    freqval = ts_freq
    filteredfile = infile[[dateval,tgtval]]
    rtval = validate(filteredfile,dateval,tgtval)
    if rtval ==1: print('Daily Data')
    elif rtval ==2: print("Weekly Data")
    elif rtval ==3: print("Monthly Data")
    elif rtval==4: print("Yearly Data")
    else: print("Need more data")
    ifile = filteredfile.set_index(dateval)
    ifile.index = pd.to_datetime(ifile.index)
    reagg = ifile
    if int(freqval)>=rtval: 
        if int(freqval)>rtval: reagg = reagg.resample(freq_dict[int(freqval)]).mean()
        reagg.to_csv(app_root+indexpath+modelname+'_'+filename)
        reagg = np.log(reagg+1)
        lmodel, trend = fit_linear(reagg.values)
        joblib.dump(lmodel,app_root+submodelpath+modelname+'_lmodel.sav')
        de_trend = remove_fit(reagg.values,trend)
        poly_deg, poly_coef, seasonal = fit_poly(de_trend,freqval)
        de_seasonal = remove_fit(de_trend,seasonal)
        dfresult = dickyfuller_test(de_seasonal.flatten())
        print('ADF Statistic: ',dfresult[0])
        print('p-value: ',dfresult[1])
        if dfresult[1]<=0.05: print('Series is Stationary')
        np.savetxt(app_root+indexpath+modelname+'_deseas_'+filename,de_seasonal,delimiter=',')
        train_info['deseasonal_file']=modelname+'_deseas_'+filename
        size = int(len(de_seasonal) * 0.80)
        train, test = de_seasonal[0:size], de_seasonal[size:len(reagg)]
        squared_error,accuracy, arima_order = get_configuration(train,test,pvalues,qvalues)
        print(squared_error, arima_order, accuracy)
        train_info['model']='arima'
        train_info['linear_model']='lmodel'
        train_info['poly_deg']=poly_deg
        train_info['poly_coef']=poly_coef.tolist()
        train_info['infreq']=freqval
        train_info['arima_order']=arima_order
        train_info['linear'] = modelname+'_lmodel.sav'
        
        js = json.dumps(train_info)
        with open(app_root+modelpath+modelname+'.json','w') as f:
            f.write(js)
        pred_values = arima_predict(train,test,arima_order, lmodel, poly_deg, poly_coef, freqval)

        fig, ax = plt.subplots(figsize=(20,6))
        ax.plot(reagg.index.values[:len(train)+1],np.exp(reagg)[:len(train)+1], color='blue', label ='Train values')
        ax.plot(reagg.index.values[len(train):],pred_values,color='red', label='Forecasted Test values')
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.set_ylabel(val_col)
        ax.set_title('Actual & Forecasted plot', fontsize=20)
        ax.legend()
        fig.savefig(app_root+imagepath+'ARIMA_Trained_plot.jpg')
        plt.close(fig)
#        plt.show()
#        plt.plot(np.exp(reagg.values))
#        plt.plot(trend)
#        plt.plot(de_trend)
#        plt.plot(seasonal)
#        plt.show()
#        
#    
#        plt.plot(de_seasonal)
#        plt.show()
        plots={'plot1':app_root+imagepath+'ARIMA_Trained_plot.jpg','score':str(round((accuracy*100),2))+'%'}
    else: print("Frequency doesnt match")
    return plots

def load_jsonfile(model,weights):
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)    
    return loaded_model

def create_lags(df,infreq):
    lag_value = freq_to_days[int(infreq)]
    for column in df.columns:
        df[column+'_'+str(lag_value)] = df[column].shift(lag_value)
    df.dropna(inplace=True)
    return df
 
def train_lstm(Xtrain,ytrain,Xvalid,yvalid):
    np.random.seed(1)
    model = Sequential()
    model.add(LSTM(300, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dropout(0.05))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(Xtrain, ytrain, epochs=40, batch_size=3,validation_data=(Xvalid, yvalid),verbose=0)
#    plt.plot(history.history['loss'], label='train')
#    plt.plot(history.history['val_loss'], label='test')
#    plt.legend()
#    plt.show()
    return model

def lstm_predict(modelname,testfile,test_holiday,ts_col,regressors):
    pred={}
    tfile = pd.read_csv(testfile)
    dateval = ts_col
    regressor = regressors
    
    if regressor is not None: regressor = regressor.split(',')
    jsmodel = json.load(open(app_root+modelpath+modelname+'.json'))
    freqval = jsmodel['infreq']   
    if test_holiday is not None: 
        hfile = pd.read_csv(test_holiday)
    else: hfile = None
    
    test, temp_dict = merge_data(tfile,hfile,dateval,None,regressor,{})
    test = test.reset_index()
    test_columns = test.columns
    
    if jsmodel['holidayfile']==1:
        train_hcols = jsmodel['holidayfields']
        for cols in train_hcols:
            if cols not in test_columns: test[cols]=0
    
    if jsmodel['other_regressors']==1:
        train_rcols = jsmodel['regressor_fields']
        for cols in train_rcols:
            if cols not in test_columns: test[cols]=0
    trainfile = jsmodel['filename']
    trainfile = pd.read_csv(app_root+indexpath+trainfile)
    new_test = pd.DataFrame()
    for cols in trainfile.columns:
        if cols in test.columns: new_test[cols]= test[cols]
        else: new_test[cols]=0

    new_test = new_test.set_index(dateval)
    new_test.index = pd.to_datetime(new_test.index)
    if int(freqval)>1: new_test = new_test.resample(freq_dict[int(freqval)]).mean()
    trainfile = trainfile.set_index(dateval)
    trainfile.index = pd.to_datetime(trainfile.index)

    trainlen = len(trainfile)
    consol_file = trainfile.append(new_test)
    consol_file = create_lags(consol_file,freqval)

    trainlen = trainlen-freq_to_days[int(freqval)]

    loaded_model = load_jsonfile(app_root+submodelpath+modelname+'_lstm_model.json', app_root+submodelpath+modelname+'_''lstm_weights.h5')
    loaded_standardizer = joblib.load(app_root+submodelpath+modelname+'_standardizer.sav')
    test_scaled = loaded_standardizer.transform(consol_file[trainlen:].values)
    test_X = test_scaled[:,1:]
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    yhat = loaded_model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.concatenate((yhat, test_X), axis=1)
    inv_yhat = loaded_standardizer.inverse_transform(inv_yhat)  
    inv_yhat = inv_yhat[:,0]
    #print(inv_yhat)
    #inv_yhat = [round(inv_yhat,2) for x in inv_yhat]
    print(len(new_test),len(inv_yhat))
    fig = plt.figure(figsize=(15,7))
    ax1 = plt.subplot(111)
    ax1.plot(new_test.index.values,inv_yhat, color='green', label="Predicted values")
    ax1.xaxis.set_tick_params(rotation=45)
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(monthsFmt)
    ax1.set_ylabel('output')
    ax1.set_title('Forecast plot', fontsize=20)
    ax1.legend()
    fig.savefig(app_root+imagepath+'LSTM_Predicted_plot.jpg')
    plt.close(fig)
    plots=app_root+imagepath+'LSTM_Predicted_plot.jpg'
    #for ts in range(len(new_test)):
    #    pred[str(new_test.index[ts-1].date())]=round(inv_yhat[ts],2)
    return plots
        
def lstm_model(modelname,filename,holidayfile,ts_col,val_col,regressors,ts_freq):
    plots ={}
    train_info={}
    infile = pd.read_csv(filename)
    filename = re.search(r'\\([^\\]+\.csv)',filename).group(0)
    filename = filename.replace('\\','')
    dateval = ts_col
    tgtval = val_col
    train_info['forecast_val']=tgtval
    regressor = regressors
    freqval = ts_freq
    if regressor is not None: regressor = [reg.strip() for reg in regressor.split(',')]
    filteredfile = infile[[dateval,tgtval]]
    rtval = validate(filteredfile,dateval,tgtval)
    
    if holidayfile is not None: 
        hfile = pd.read_csv(holidayfile)
    else: hfile = None
    ifile,train_info = merge_data(infile, hfile, dateval, tgtval, regressor,train_info)
    reagg = ifile.copy()
    #ifile['Sales'].hist()
    #plt.show()
    if int(freqval)>=rtval: 
        if int(freqval)>rtval: reagg = reagg.resample(freq_dict[int(freqval)]).mean()
        reagg.to_csv(app_root+indexpath+modelname+'_'+filename)
        train_info['model']='nn'
        train_info['filename']=modelname+'_'+filename
        train_info['infreq']=freqval
        lag_df = create_lags(reagg,freqval)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(lag_df.values)
        size = int(len(scaled_values) * 0.80)
        X_train, X_test = scaled_values[:size,1:], scaled_values[size:,1:]
        y_train, y_test = scaled_values[:size,0], scaled_values[size:,0]
        train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        lstmmodel = train_lstm(train_X,y_train,test_X,y_test)
        
        yhat = lstmmodel.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        inv_yhat = np.concatenate((yhat, test_X), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        
        y_test = y_test.reshape((len(y_test), 1))
        inv_y = np.concatenate((y_test, test_X), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        accuracy = r2_score(inv_y,inv_yhat)
        print('Test RMSE: %.3f' % rmse)
        js = json.dumps(train_info)
        with open(app_root+modelpath+modelname+'.json','w') as f:
            f.write(js)
        joblib.dump(scaler,app_root+submodelpath+modelname+'_standardizer.sav')
        lstm_model = lstmmodel.to_json()
        with open(app_root+submodelpath+modelname+'_lstm_model.json','w') as f:
            f.write(lstm_model)
        lstmmodel.save_weights(app_root+submodelpath+modelname+'_lstm_weights.h5')
        fig = plt.figure(figsize=(15,7))
        ax1 = plt.subplot(111)
        ax3 = plt.subplot(111)		
        ax4 = plt.subplot(111)
        #print(reagg.index.values)
        #ax1.plot(reagg.index.values[:size+1],lag_df[tgtval][:size+1], color='blue', label="Actual values")
        ax1.plot(reagg.index.values,lag_df[tgtval], color='blue', label="Actual values")
        ax4.plot(reagg.index.values[size:],inv_yhat,color='green', label="Forecasted values", marker='o', ls='None')
        stderr = abs(lag_df[tgtval][size:]-inv_yhat)
        dev_level = stderr.mean()+stderr.std()
        print(stderr.mean(),stderr.std(), dev_level)
        #print(ifile.head())
        is_dev=stderr.apply(lambda x: True if x>dev_level else False).values
        #print(is_dev)
        #print(stderr[is_dev])
        print(lag_df[tgtval][size:][is_dev])
        anomaly_df = ifile.loc[lag_df[tgtval][size:][is_dev].index.values]
        anomaly_df.index=anomaly_df.index.astype(str)
        anomaly_array = ast.literal_eval(anomaly_df.reset_index().to_json(orient='values'))
        #print(type(list(anomaly_df.columns.values)),type(anomaly_array))
        anomaly_array=[list(anomaly_df.reset_index().columns.values)]+anomaly_array
        with open(app_root+anomaly+filename.split('.')[0]+'.json','w') as f:
            f.write(str(anomaly_array))
        
        ax3.plot(stderr[is_dev].index.values,lag_df[tgtval][size:][is_dev],color='red', label="Anomaly (Actual value)",ls='None',marker='*')
        #anomaly = stderr.loc[stderr['is_dev']==1]
        #print(anomaly)
        ax1.xaxis.set_tick_params(rotation=45)
        ax1.xaxis.set_major_locator(months)
        ax1.xaxis.set_major_formatter(monthsFmt)
        ax1.set_ylabel(val_col)
        ax1.set_title('Actual & Forecasted plot', fontsize=20)
        ax1.legend()
        ax4.legend()
        fig.savefig(app_root+imagepath+'LSTM_Trained_plot.jpg')
        plt.close(fig)
        plots={'plot1':app_root+imagepath+'LSTM_Trained_plot.jpg', 'score':str(round((accuracy*100),2))+'%'}
    else: print("Frequency doesnt match")

    return plots


def ts_train(model,modelname,file,holidayfile,datefield,tgtfield,regfield,frequency):
    freq = str_to_freq[frequency.lower()]
    rt_plots={}
    if model.lower()=='arima':
        rt_plots = train_arima(modelname,file,datefield,tgtfield,freq)
    elif model.lower()=='nn':
        rt_plots = lstm_model(modelname,file,holidayfile,datefield,tgtfield,regfield,freq)
    return rt_plots

def ts_predict(modelname,tfile,thfile,datefield,regfield):
    jsonmodel = json.load(open(app_root+modelpath+modelname+'.json'))
    output={}
    model = jsonmodel['model']
    if model=='arima':
        pvalues = test_arima(modelname,tfile,datefield)
        #pvalues = np.squeeze(pvalues)
        
    elif model=='nn':
        pvalues = lstm_predict(modelname,tfile,thfile,datefield,regfield)
    #output['prediction']=OrderedDict(sorted(pvalues.items(), key=lambda t:t[0]))
    output['prediction']=pvalues
    return output

#ps = summarize('Storedata.csv','Date','Sales',2)  
#ps = summarize('shampoo_sales.csv','Month','Sales',3)  
#ps = summarize('AirPassengers.csv','Month','Passengers',3)                        
#print(ps)

#pt = ts_train('NN','storelstm','Storedata.csv','holiday_file.csv','Date','Sales','Open,Promo',2)
#pt = ts_train('NN','shampoolstm','shampoo_sales.csv',None,'Month','Sales',None,3)
#pt = ts_train('NN','passengerlstm','AirPassengers.csv',None,'Month','Passengers',None,3)
#print(pt)

#pval = ts_predict('storelstm','testfile.csv','testholiday.csv','Date','Open,Promo')
#pval = ts_predict('shampoolstm','test_shampoo.csv',None,'Month',None)
#pval = ts_predict('passengerlstm','test_passengers.csv',None,'Month',None)
#print(pval)

#pt = ts_train('arima','storearima','Storedata.csv','holiday_file.csv','Date','Sales','Open,Promo',2)
#pt = ts_train('arima','shampooarima','shampoo_sales.csv',None,'Month','Sales',None,3)
#pt = ts_train('arima','passengerarima','AirPassengers.csv',None,'Month','Passengers',None,3)
#print(pt)

#pval = ts_predict('storearima','testfile.csv','testholiday.csv','Date','Open,Promo')
#pval = ts_predict('shampooarima','test_shampoo.csv',None,'Month',None)
#pval = ts_predict('passengerarima','test_passengers.csv',None,'Month',None)
#print(pval)

