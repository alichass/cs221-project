from predutil import *
import pandas as pd

company = 'Google'
sentimentcsv = 'data/' +company + '/sentiment/'  +company.lower() + 'foolsentiment.csv'
stockname = 'GOOG'
stockcsv = 'data/' + company + '/' + stockname + '.csv'
trendcsv = 'data/' +company+ '/' + 'multiTimeline.csv'

trendData = pd.read_csv(sentimentcsv)
trendMap = createSentimentMap(trendData, niave=True, coded=True)
print trendMap['2017-08-18']
