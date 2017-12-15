import numpy as np
import math
import datetime
import collections
from datetime import timedelta
from dateutil.parser import parse

def dateToWeek(date):
    #datetime.date(2010, 6, 16).isocalendar()[1]
    
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    
    weekNum = datetime.date(year, month, day).isocalendar()[1]
    
    return str(year) + '-' + str(weekNum)

def createWeeklyTrendMap(trenddata):
    weekmap = {}
    trendData = trenddata.values
    for day_trend in trendData:
        trendval = day_trend[1]
        dt = parse(day_trend[0])
        for i in range(7):
            weekmap[(dt + timedelta(days=i)).strftime('%Y-%m-%d')] = float(trendval)/100
    return weekmap


def createSentimentMap(sentimentdata, niave=True, coded=False):
    if niave == False:
        coded = False
    sentimentmap = {}
    sentimentDataValues = sentimentdata.values
    if niave == False:
        for day_sentiment in sentimentDataValues:
            sentvals = day_sentiment[1:4]
            dt = parse(day_sentiment[0])
            sentimentmap[dt.strftime('%Y-%m-%d')] = sentvals
        return sentimentmap
    if niave == True:
        for day_sentiment in sentimentDataValues:
            sentvals = day_sentiment[1:4]
            index = np.argmax(sentvals)
            dt = parse(day_sentiment[0])
            if index == 0:
                sentimentmap[dt.strftime('%Y-%m-%d')] = 0.0
            if index == 1:
                if coded == True:
                    sentimentmap[dt.strftime('%Y-%m-%d')] = math.floor(sentvals[index] * -3)
                else:
                    sentimentmap[dt.strftime('%Y-%m-%d')] = -1.0
            if index == 2:
                if coded == True:
                    sentimentmap[dt.strftime('%Y-%m-%d')] = math.ceil(sentvals[index] * 3)
                else:
                    sentimentmap[dt.strftime('%Y-%m-%d')] = 1.0
        return sentimentmap



def CreateAvrgWeeklySentimentMap(sentimentdata):
    sentimentDataValues = sentimentdata.values

    groupbyweek = collections.defaultdict(lambda: np.ndarray((1)))
    weekcount = collections.defaultdict(int)
    for day_sentiment in sentimentDataValues:
        dt = parse(day_sentiment[0])
        week_year =  str(dt.isocalendar()[0]) +'-' + str(dt.isocalendar()[1])
        groupbyweek[week_year] = day_sentiment[1:4] + groupbyweek[dt.isocalendar()[1]]
        weekcount[week_year]+=1
    for week in groupbyweek:
        groupbyweek[week]= groupbyweek[week]/weekcount[week]
    sentimentmap = {}

    for day_sentiment in sentimentDataValues:
        dt = parse(day_sentiment[0])
        week_year =  str(dt.isocalendar()[0]) +'-' + str(dt.isocalendar()[1])
        sentvals = groupbyweek[week_year]
        sentimentmap[dt.strftime('%Y-%m-%d')] = sentvals

    return sentimentmap

def parseSentimentMonthly(sentimentDataValues):

    # get week list
    months = []
    for i in range(len(sentimentDataValues)):
        curRow = sentimentDataValues[i, :]
        curDate = curRow[0]
        curMonth = curDate[0:7]
        months.append(curWeek)
    months = set(months)

    # prepare week map for value entry
    # index 3 is for counts
    monthMap = {}
    for month in months:
        values = [0.0, 0.0, 0.0, 0]
        monthMap[month] = values

    # add values from relevant dates to the corresponding week in the map
    for i in range(len(sentimentDataValues)):
        curRow = sentimentDataValues[i, :]
        curDate = curRow[0]
        curMonth = curDate[0:7]

        monthMap[curMonth][3] += 1

        for j in range(3):
            curSentimentVal = curRow[j + 1]
            monthMap[curMonth][j] += curSentimentVal

    # take the average of each value
    for month in monthMap:
        denominator = monthMap[month][3]
        for i in range(3):
            monthMap[month][i] = (monthMap[month][i] / denominator)

    return monthMap


def dateToWeek(date):
    #datetime.date(2010, 6, 16).isocalendar()[1]

    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])

    weekNum = datetime.date(year, month, day).isocalendar()[1]

    return str(year) + '-' + str(weekNum)






#
# # place sentiment on correct days
# sentimentDates = list(sentimentDataValues[:, 0])
# for i in range(len(training_set_scaled)):
#     curDate = training_set_scaled[i, 5]
#
#     if curDate in sentimentDates:
#
#         index = sentimentDates.index(curDate)
#         curRow = sentimentDataValues[index, :]
#
#         for j in range(3):
#             curSentiment = curRow[j + 1]
#             training_set_scaled[i, j + 2] = curSentiment
#
#
# #for i in range(len(training_set_scaled)):
#  #   print training_set_scaled[i, :]
#
# # distribute sentiment temporally
#
# changed = set()
# for i in range(len(training_set_scaled)):
#     curRow = training_set_scaled[i, :]
#     curSentiment = curRow[2]
#
#     if curSentiment != 0 and i not in changed:
#         # spread it forward by 5 days, or until you reach a different val
#
#         changeCount = 0
#         for j in range(i + 1, i + 3):
#             if j < len(training_set_scaled) - 1:
#
#                 nextRow = training_set_scaled[j, :]
#                 nextSentiment = nextRow[2]
#
#                 if nextSentiment == 0:
#                     changed.add(j)
#                     training_set_scaled[j, 2] = curSentiment
#                 else:
#                     break
#


    # add trend data
    #
    # newData = np.zeros((len(training_set_scaled), 1))
    # training_set_scaled = np.append(training_set_scaled, newData, axis=1)
    #
    # trendAverage = 0.5
    #
    # for i in range(len(training_set_scaled)):
    #     curDate = training_set_scaled[i, len(training_set_scaled[i,:]) - 2]
    #     curWeek = dateToWeek(curDate)
    #
    #     if curWeek in trendMap:
    #         curTrend = trendMap[curWeek][0]
    #         training_set_scaled[i, 6] = curTrend
    #
    #     else:
    #         training_set_scaled[i, 6]  = trendAverage
