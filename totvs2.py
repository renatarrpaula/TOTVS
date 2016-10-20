from sys import argv
import json
import scipy
import scipy.sparse as ss
import datetime
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.ensemble.forest import RandomForestRegressor
#from clearall import clear_all
import matplotlib.pyplot as plt
from os.path import exists

#load data into a list
jsonfile = open("sample.json")
data = json.load(jsonfile)

# Number of data points
n = len(data)

# Existing items that can be bought
nitems = [None]*n
items = list()
for i in xrange(0,n):
    nitems[i] = len(data[i]['dets'])
    for j in xrange (0, nitems[i]-1):
        if data[i]['dets'][j]['prod']['xProd'] not in items:
            items.append(data[i]['dets'][j]['prod']['xProd'])

#After identified, items sorted by type
food = ['TEMAKI', 'SUSHI ESPECIAL', 'SASHIMI', 'YAKISSOBA', 'URUMAKI']
beverages = ['AGUA', 'SUCO', 'CHA', 'LIMONADA', 'REFRIGERANTE']
alcohol = ['CERVEJA', 'SAKE', 'CERVEJA LATA', 'WHISKY', 'CAIPIROSKA', 'CAIPIRINHA', 'BACARDI']
dessert = ['SOBREMESA']
coffe = ['CAFE EXPRESSO']

# Estimated most important features for predictions: 
    #date, weekdays, hour, number of different items per category
dateobj = [None]*n
weekd = [None]*n

#Categories of products
nfood = [0]*n
nbeverages = [0]*n
nalcohol = [0]*n
ndessert = [0]*n
ncoffe = [0]*n
nbuffet = [0]*n

for i in xrange(0,n):
    # Create datetime object and weekday list
    dateobj[i] = datetime.datetime.strptime(data[i]['ide']['dhEmi']['$date'],'%Y-%m-%dT%H:%M:%S.%fZ')
    weekd[i] = dateobj[i].weekday()
    
    # Amount of items bought
    nitems[i] = len(data[i]['dets'])
        
    for j in xrange (0, nitems[i]):
        
        # what items? How many different items per category?
        if data[i]['dets'][j]['prod']['xProd'] in 'BUFFET':
            nbuffet[i] += 1*data[i]['dets'][j]['prod']['qCom']
        if data[i]['dets'][j]['prod']['xProd'] in food:
            nfood[i] += 1*data[i]['dets'][j]['prod']['qCom']
        if data[i]['dets'][j]['prod']['xProd'] in beverages:
            nbeverages[i] += 1*data[i]['dets'][j]['prod']['qCom']
        if data[i]['dets'][j]['prod']['xProd'] in alcohol:
            nalcohol[i] += 1*data[i]['dets'][j]['prod']['qCom']
        if data[i]['dets'][j]['prod']['xProd'] in dessert:
            ndessert[i] += 1*data[i]['dets'][j]['prod']['qCom']
        if data[i]['dets'][j]['prod']['xProd'] in coffe:
            ncoffe[i] += 1*data[i]['dets'][j]['prod']['qCom']

# We can estimate one client expenses just by the amount of different items 
# per category (nfood, nbeverages, etc...) and the weekday and hour of the day
features = [list()]*n
value = [float()]*n
for i in xrange(0,n):
    features[i] = [weekd[i], dateobj[i].hour, nbuffet[i], nfood[i], nbeverages[i], nalcohol[i], ndessert[i], ncoffe[i]]
    value[i] = data[i]['complemento']['valorTotal']


# Multiple linear regression
# Fits the models in 90% of the data and tests it in the last 10%
Xtrain = features[0:int(round(0.9*len(features)))]
ytrain = value[0:int(round(0.9*len(features)))]
Xtest = features[int(round(0.9*len(features))):len(features)]
ytest = value[int(round(0.9*len(features))):len(features)]

#Train the model and predict for test set
#Variables were not scaled because they have about the same order of magnitude
clf = linear_model.LinearRegression()
cf = clf.fit(Xtrain,ytrain)
s = clf.score(Xtest,ytest)
ycalc = clf.predict(Xtest)

# Plot results for test set
plt.plot(ytest, ycalc,'.')
ytest = numpy.array(ytest)
m, b = numpy.polyfit(ytest, ycalc, 1)
plt.plot(ytest, m*ytest + b, '-')
plt.ylabel('Predicted expenses per customer')
plt.xlabel('Actual expenses per customer')
plt.show()

print s, clf.coef_

#Calculate number of weeks in the dataset and sales per week
numberofweeks = (dateobj[n-1].day-dateobj[0].day)/7 + 1
expensesweek = [0]*numberofweeks
i = 0
j = 0
for i in xrange(0,n):
    if i == (n-1):
        expensesweek[numberofweeks-1] = expensesweek[numberofweeks-1] + value[i]
    elif weekd[i+1] >= weekd[i]:
        expensesweek[j] = expensesweek[j] + value[i]
    else:
        j += 1
        expensesweek[j] = expensesweek[j] + value[i]

# Function
def func(x, p1,p2,p3):
  return p1*x**2+p2*x+p3

weeks = numpy.array(range(0,numberofweeks))
popt, pcov = curve_fit(func, weeks , expensesweek, p0 = (-1,1,24000))

#curve params
p1 = popt[0]
p2 = popt[1]
p3 = popt[2]

#Plot sales per week
plt.plot(weeks, expensesweek,'.')
curve=func(numpy.linspace(0,2,100),p1,p2,p3)
plt.plot(numpy.linspace(0,2,100),curve,'r', linewidth=2)
plt.ylabel('Sales')
plt.xlabel('Weeks')


