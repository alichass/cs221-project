# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import os
import datetime
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.layers.core import Activation
from sklearn.preprocessing import MinMaxScaler
from predutil import *
import matplotlib.pyplot as plt


def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

saveResults = True
testSetSize = 35
daysBack = 5

directoryName = 'ModelOutputs'
createDirectory(directoryName)

company = 'Apple'
sentimentcsv = 'data/' +company + '/sentiment/'  +company.lower() + 'foolsentiment.csv'
stockname = 'AAPL'
stockcsv = 'data/' + company + '/' + stockname + '.csv'
trendcsv = 'data/' +company+ '/' + 'multiTimeline.csv'

paramMaster=[]

# add a nonsentiment,plain historical value one
# weekly avg, 
paramMaster.append([0,0,0,1,0,'noVolume_noHighLow_Niave_noCoded_noTrend',0])
paramMaster.append([1,1,1,1,0,'Volume_HighLow_Niave_noCoded_noTrend',0])
paramMaster.append([0,0,1,1,0,'noVolume_HighLow_Niave_noCoded_noTrend',0])
paramMaster.append([0,0,0,1,0,'noVolume_noHighLow_Niave_noCoded_Trend',1])
paramMaster.append([0,0,1,1,0,'noVolume_HighLow_Niave_noCoded_Trend',1])
paramMaster.append([0,0,1,1,1,'noVolume_HighLow_Niave_Coded_Trend',1])
paramMaster.append([0,0,1,0,0,'noVolume_HighLow_noNiave_noCoded_Trend',0])
paramMaster.append([0,0,1,1,1,'noVolume_HighLow_Niave_Coded_noTrend',0])
paramMaster.append([0,0,1,0,0,'noVolume_HighLow_noNiave_noCoded_noTrend',0])

paramMaster2=[]

paramMaster2.append([0,0,1,0,0,'noVolume_HighLow_noNiave_noCoded_Trend_weeklyAvg',1, 1, 0])
#paramMaster2.append([0,0,0,0,0,'noVolume_noHighLow_noNiave_noCoded_noTrend_baseline',0, 0, 1])
#paramMaster2.append([0,0,1,0,0,'noVolume_HighLow_noNiave_noCoded_noTrend_baseline',0, 0, 1])


def dostuff(params):
        #settings for stock data
    useVolume = params[0]
    scaleVolume = params[1]
    useHighLow=params[2] 
    
    #settings for sentiment data
    niave=params[3]
    coded=params[4]
    
    graphpath = 'ModelOutputs/'+ company+ '_' +params[5]
    print graphpath
    
    def preprocessData(currentStockID, otherdata, daysBack=daysBack, useVolume = useVolume, scaleVolume = scaleVolume, useHighLow=useHighLow):
        if useVolume == False:
            scaleVolume = False
    
        # open historical dataa
        stockData = pd.read_csv(stockcsv)
        stockData = stockData.iloc[-470:,:]
        # extract the date column
        dates = stockData.iloc[:, 0].values
        dates = np.reshape(dates, (len(dates), 1))
    
        # Extract the rest of the stock-price data
        training_set = stockData.iloc[:,1:5].values
    
        # Delete unneeded data based on user request
        if useHighLow == False:
            training_set = np.delete(training_set,[1,2], axis=1)
    
        # Scale stock-price data
        flatprices = training_set.flatten()
        flatprices = np.reshape(flatprices, (len(flatprices),1))
        sc = MinMaxScaler(feature_range = (0, 1))
        sc.fit(flatprices)
        training_set_scaled = sc.transform(training_set)
    
        #if requested add the volume column to the training_set
        if useVolume == True:
            volume = stockData.iloc[:, 6].values
            volume = np.reshape(volume, (len(volume), 1))
            if scaleVolume == True:
                volumeScaler = MinMaxScaler(feature_range = (0,1))
                volume = volumeScaler.fit_transform(volume)
            training_set_scaled = np.append(training_set_scaled, volume, axis=1)
    
    
        #add the date column to the training data in order to add otherdata
        training_set_scaled = np.append(dates, training_set_scaled, axis=1)
    
        # # add columns for Other values and set those values
        for dataset in otherdata:
            numnewcols = 1
            if not isinstance(dataset.values()[0], float):
                numnewcols = len(dataset.values()[0])
            newcol = np.zeros((len(training_set_scaled), numnewcols))
            training_set_scaled = np.append(training_set_scaled, newcol, axis=1)
            for i in range(len(training_set_scaled)):
                if np.any(dataset.get(training_set_scaled[i,0]) != None) :
                    training_set_scaled[i,training_set_scaled.shape[1] - numnewcols:training_set_scaled.shape[1]] = dataset[training_set_scaled[i,0]]
    
        # delete date column
        training_set_scaled = np.delete(training_set_scaled,[0], axis=1)
    
    
        # Split the test data away from the training data
        trainBound = len(stockData) - testSetSize
    
        test_data = training_set_scaled[trainBound:]
        training_set_scaled = training_set_scaled[0:trainBound]
        
    
        # Creating a data structure with  timesteps and t+1 output
        X_train = []
        y_train = []
        for i in range(daysBack, trainBound - 1):
            X_train.append(training_set_scaled[i-daysBack:i, 0:])
            y_train.append(training_set_scaled[i, 1])
        X_train, y_train = np.array(X_train), np.array(y_train)
        # Reshaping
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
        return (X_train, y_train, training_set_scaled, test_data, sc)
    
    def createModel(sz):
        model = Sequential()
    
        # layer architecture
        model.add(GRU(units = 512,input_shape = (None, sz), return_sequences=True,recurrent_dropout = 0.25))
        model.add(Dropout(0.2))
        model.add(GRU(units = 512,recurrent_dropout = 0.25))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))
        # Compiling
        model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    
        return model
    
    def baselinePredict(sc, real_data):
        opens = real_data[:, 1]
        predictions = []
        for i in range(1, len(opens)):
            prevPrice = opens[i-1]
            curPrice = opens[i]
            change = curPrice - prevPrice
            tomorrow = curPrice + change
            predictions.append(tomorrow)
    
        predictions = predictions[5:]
        predictions = np.array(predictions)
    
        predictions = np.reshape(predictions, (len(predictions),1))
        predictions = sc.inverse_transform(predictions)
        return predictions
    
    def getPredictions(model, sc, sz, real_data):
        inputs = []
        for i in range(daysBack, len(real_data) - 1):
            inputs.append(real_data[i-daysBack:i, 0:])
    
        inputs = np.array(inputs)
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], sz))
    
        predicted_stock_price = model.predict(inputs)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
        return predicted_stock_price
    
    def invertRealPrices(real_data):
        real_stock_price = sc.inverse_transform(real_data)
        real_stock_price = real_stock_price[daysBack:, 1]
    
        return real_stock_price
    
    def displayAndSaveResults(save, realStockPrices, predictedStockPrice, currentStockID, graphPath):
         realLabel = 'Real ' +  currentStockID + ' Stock Price'
         predictedLabel = 'Predicted ' +  currentStockID + ' Stock Price'
         titleLabel = currentStockID + ' Stock Price Prediction'
         yLabel = currentStockID + ' Price'
         plt.plot(realStockPrices, color = 'red', label = realLabel)
         plt.plot(predictedStockPrice, color = 'blue', label = predictedLabel)
         plt.title(titleLabel)
         plt.xlabel('Time')      
         plt.ylabel(yLabel)
          
         plt.legend()
         
    
         print "norm"
    
         realChanges = [(realStockPrices[i+1]-realStockPrices[i])/realStockPrices[i] for i in range(len(realStockPrices) - 1)]
         predictedChanges = [(predictedStockPrice[i+1]-predictedStockPrice[i])/predictedStockPrice[i] for i in range(len(predictedStockPrice) - 1)]
    
         thisnorm = np.linalg.norm(np.array(realChanges) - np.array(predictedChanges))
    
         predictedRightDirection = [abs(i+j) == abs(i) + abs(j) for i, j in zip(realChanges, predictedChanges)]
         thismean = np.mean(predictedRightDirection) * 100
         annotation = '2-Norm Distance: ' + str(thisnorm) +'\n' + '% Right: ' +str(thismean)
         
         plt.annotate(annotation, xy=(0.01, 0.6), xycoords='axes fraction')
         
    
         if save:
             plt.savefig(graphPath)
    
         plt.show()
         
         
         
    
    print(company)
    
    
    # open sentimnet data
    sentimentData = pd.read_csv(sentimentcsv)
    sentimentmap = createSentimentMap(sentimentData, niave=niave, coded=coded)
    #open trend data
    trendData = pd.read_csv(trendcsv)
    trendMap = createWeeklyTrendMap(trendData)
    
    weeklyAverage=CreateAvrgWeeklySentimentMap(sentimentData)
    otherdata=[]
    #process the data
    if params[7]==1:
        otherdata=[weeklyAverage]
    else:
        otherdata=[sentimentmap]
        
        
    if params[6] ==1:
        otherdata.append(trendMap)
        
    if params[8]==1:
        otherdata=[]
        
    

        
    X_train, y_train, training_set_scaled, test_data, sc = preprocessData(stockname, otherdata)
    print "Sample Data point:"
    print training_set_scaled[-50:-45]
    
    #create and fit the model 
    model = createModel(training_set_scaled.shape[1])
    model.fit(X_train, y_train, epochs = 100, batch_size = 32, verbose=True)
    
    #predict stock prices using this model
    predicted_stock_price = getPredictions(model, sc, training_set_scaled.shape[1], test_data)
    
    #predict stock prices using a baseline model
    #baseline_stock_price = baselinePredict(sc, test_data)
    
    # the real price of these stocks
    real_stock_price = invertRealPrices(test_data)
    
    displayAndSaveResults(True, real_stock_price, predicted_stock_price, stockname, graphpath)

#print predicted_stock_price
#print baseline_stock_price
#print real_stock_price
companies=[('Nvidia','NVDA'),('Google','GOOG'),('EA','EA'),('Apple','AAPL')]

for c in companies:
    for i in paramMaster2:
    
    
        company = c[0]
        sentimentcsv = 'data/' +company + '/sentiment/'  +company.lower() + 'foolsentiment.csv'
        stockname = c[1]
        stockcsv = 'data/' + company + '/' + stockname + '.csv'
        trendcsv = 'data/' +company+ '/' + 'multiTimeline.csv'
        
        dostuff(i)


#
#
#
#
# #datetime.date(2010, 6, 16).isocalendar()[1]
