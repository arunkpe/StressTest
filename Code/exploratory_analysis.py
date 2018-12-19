import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as smodels
import statsmodels.tsa as smt
import readData as rd
import os
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor

dir_path = os.getcwd()

cmbsSpread = pd.read_excel(os.path.join(dir_path,'CMBSSpreads.xlsx'),sheet=0)
cmbsSpread = cmbsSpread.drop(cmbsSpread.index[0])
cmbsSpread.rename(columns=lambda x: x.strip(),inplace = True)
cmbsSpread.columns = ['Date','Senior','Mezz','Junior']
cmbsSpread['LastDate'] = cmbsSpread['Date']
cmbsSpread.set_index('Date',inplace = True)
cmbsSpread = cmbsSpread.loc[cmbsSpread.groupby(pd.TimeGrouper('M')).idxmax().iloc[:,3]]
cmbsSpread.reset_index(inplace = True)
cmbsSpread['Date'] = cmbsSpread['Date'].dt.date
cmbsSpread = cmbsSpread.drop('LastDate',axis=1)
bonds = ['Senior','Mezz','Junior']
for var in bonds:
    cmbsSpread[var] = ((cmbsSpread[var]/100).astype(float))

bondTranche = 'Mezz'
#First Order Difference
autoCorr = 'Mezz_lag1'
autoCorrflg = 0

rolmean = pd.Series.rolling(cmbsSpread[bondTranche], window=3)
rolstd = pd.Series.rolling(cmbsSpread[bondTranche], window=3)
test_stationarity(cmbsSpread[bondTranche].astype(float))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = smodels.graphics.tsa.plot_acf(cmbsSpread[bondTranche].astype(float), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = smodels.graphics.tsa.plot_pacf(cmbsSpread[bondTranche].astype(float), lags=40, ax=ax2)



#Explore TS
igSpreadExplore = cmbsSpread[['Date',bondTranche]]
igSpreadExplore[bondTranche] = pd.to_numeric(igSpreadExplore[bondTranche])
igSpreadExplore['Date'] = pd.to_datetime(igSpreadExplore['Date'])
igSpreadExplore =igSpreadExplore.set_index('Date')
smt.seasonal.seasonal_decompose(igSpreadExplore,freq=4).plot()

igSpreadts = cmbsSpread[['Date',bondTranche]]
igSpreadts['SpreadLag1'] =cmbsSpread[bondTranche].shift(2) - cmbsSpread[bondTranche].shift(1)
igSpreadts = igSpreadts.dropna()
test_stationarity(igSpreadts['SpreadLag1'].astype(float))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = smodels.graphics.tsa.plot_acf(igSpreadts['SpreadLag1'].astype(float), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = smodels.graphics.tsa.plot_pacf(igSpreadts['SpreadLag1'].astype(float), lags=40, ax=ax2)

#Explore First Order diff TS
igSpreadts1 = igSpreadts[['Daily','SpreadLag1']]
igSpreadts1['SpreadLag1'] = pd.to_numeric(igSpreadts1['SpreadLag1'])
igSpreadts1 = igSpreadts1.set_index('Daily')
smt.seasonal.seasonal_decompose(igSpreadts1,freq=52).plot()


xlsFile = pd.ExcelFile(os.path.join(dir_path,'Econ.xlsx'),sheet=1)
vixDataRaw = xlsFile.parse('VIX data')
vixData = vixDataRaw[[0,1]][1:206]
vixData.rename(columns=lambda x: x.strip(),inplace = True)

fig, ax = plt.subplots()
ax2 = ax.twinx()
vixData.plot(x='Mthly', y='VIX',color ='red',ax=ax,legend = True)
cmbsSpread.plot(x='Date',y=bondTranche,color ='blue',ax=ax2,legend = False)
cmbsSpread.plot(x='Date',y='Mezz',color ='green',ax=ax2,legend = False)
cmbsSpread.plot(x='Date',y='Junior',color ='pink',ax=ax2,legend = False)
ax.set_title("CMBS Spread and VIX")


#Read Data
readFile = rd.readData('MoodysCMT.txt')
macroVarName = readFile['macroVarName']
geoLocation  = readFile['geoLocation']
macroData    = readFile['macroData']
macroData    = macroData.replace('ND', 0)
macroData    = macroData[0:252]

bbbSpread = getMacroVar('BBB Corporate Bond', macroVarName, geoLocation, macroData)
treas10  = getMacroVar('10 Year Constant Maturity',macroVarName,geoLocation,macroData)
treas2   = getMacroVar('2 Year Constant Maturity',macroVarName,geoLocation,macroData)
treas5   = getMacroVar('5 Year Constant Maturity',macroVarName,geoLocation,macroData)
libor3   = getMacroVar('3 Month Libor',macroVarName,geoLocation,macroData)
libor6   = getMacroVar('6 Month Libor',macroVarName,geoLocation,macroData)
libor12  = getMacroVar('1 Year Libor',macroVarName,geoLocation,macroData)
fedFunds = getMacroVar('Federal Funds Rate',macroVarName,geoLocation,macroData)
treas3M  = getMacroVar('3 Month Treasury',macroVarName,geoLocation,macroData)
commHPI  = getMacroVar('Commercial Property Price Index',macroVarName,geoLocation,macroData)
treasuryCMT = treas10.set_index('Date').join(treas2.set_index('Date')).join(treas5.set_index('Date')).join(libor3.set_index('Date')).join(libor6.set_index('Date')).join(libor12.set_index('Date')).join(treas3M.set_index('Date')).join(fedFunds.set_index('Date')).join(commHPI.set_index('Date')).join(bbbSpread.set_index('Date'))
treasuryCMT.reset_index(inplace=True)
treasuryCMT.columns = ['Date','CMT10Year','CMT2Year','CMT5Year','Libor3M','Libor6M','Libor1Y','CMT3M','FedFunds','commHPI','bbbCorp']
treasuryCMT['Date'] = treasuryCMT['Date'].dt.date

djiaIdx = pd.read_excel(os.path.join(dir_path,'DJIA.xlsx'),sheet=0)
djiaIdx['Date'] = (pd.to_datetime(djiaIdx['Date']))  + pd.offsets.MonthEnd(0)
djiaIdx = djiaIdx.sort_values(by=['Date'], ascending=[1])
djiaIdx['Date'] = djiaIdx['Date'].dt.date
djiaIdx['DJIA'] = np.log(djiaIdx['DJIA'])

cmbsSpread['Date'] = pd.to_datetime(cmbsSpread['Date'])+ pd.offsets.MonthEnd(0)
vixData['Date'] = pd.to_datetime(vixData['Mthly'])+ pd.offsets.MonthEnd(0)

bondData = pd.merge(vixData, cmbsSpread, right_on='Date',left_on='Date')
bondData['VIX'] = bondData['VIX'].astype(float)
bondData['Date'] = bondData['Date'].dt.date + pd.offsets.MonthEnd(0)
bondData['Date'] = bondData['Date'].dt.date
bondSpread = pd.merge(treasuryCMT, bondData, on='Date')
bondSpread = pd.merge(djiaIdx, bondSpread, on='Date')
bondSpread.drop(['Mthly'],axis=1,inplace = True)

#Variable Transformations
bondSpread['Slope'] = bondSpread['CMT10Year'] - bondSpread['CMT2Year']
bondSpread['deltaSpread']     = bondSpread[bondTranche].shift(1) - bondSpread[bondTranche]
bondSpread['deltaSpreadOverT'] = bondSpread[bondTranche] - bondSpread['CMT10Year']
bondSpread['deltaTreas']      = bondSpread['CMT10Year'].shift(2) - bondSpread['CMT10Year'].shift(1)
bondSpread[bondTranche+'_lag1'] = bondSpread[bondTranche].shift(2) - bondSpread[bondTranche].shift(1)
bondSpread[bondTranche+'_lag2'] = bondSpread[bondTranche].shift(3) - bondSpread[bondTranche].shift(2)
bondSpread['VIX_lag1']        = bondSpread['VIX'].shift(2) - bondSpread['VIX'].shift(1)
bondSpread['VIX_lag2']        = bondSpread['VIX'].shift(3) - bondSpread['VIX'].shift(2)
bondSpread['deltaVIX']        = bondSpread['VIX'].shift(1) - bondSpread['VIX']
bondSpread['deltaBBB']     = bondSpread['bbbCorp'].shift(1) - bondSpread['bbbCorp']
bondSpread['spreadBBB']    = bondSpread['bbbCorp'] - bondSpread['CMT5Year']
bondSpread['tedSpread']    = bondSpread['Libor3M'] - bondSpread['CMT3M']
bondSpread['deltaTreas3M'] = bondSpread['CMT3M'].shift(1)-bondSpread['CMT3M']
bondSpread['deltaLibor3M'] = bondSpread['Libor3M'].shift(1)-bondSpread['Libor3M']
bondSpread['deltaLibor6M'] = bondSpread['Libor6M'].shift(1)-bondSpread['Libor6M']
bondSpread['deltaLibor1Y'] = bondSpread['Libor1Y'].shift(1)-bondSpread['Libor1Y']
bondSpread['bbbIGSpread']     = bondSpread['bbbCorp'] - bondSpread[bondTranche] - bondSpread['CMT10Year']
bondSpread['deltadjia']     = bondSpread['DJIA'].shift(2) - bondSpread['DJIA'].shift(1)
bondSpread['deltaFedFunds']     = bondSpread['FedFunds'].shift(1) - bondSpread['FedFunds']
bondSpread['fedSpread']     = (bondSpread['Libor3M'] - bondSpread['FedFunds'])
bondSpread['deltaCREIdx']     = bondSpread['commHPI'].shift(1) - bondSpread['commHPI']

#Other Variables Created
bondSpread['deltaspreadBBB']    = bondSpread['spreadBBB'].shift(1) - bondSpread['spreadBBB']
bondSpread['deltaTedSpread']    = bondSpread['tedSpread'].shift(1) - bondSpread['tedSpread']
bondSpread['deltabbbIGSpread']  = bondSpread['bbbIGSpread'].shift(1) - bondSpread['bbbIGSpread']
bondSpread['deltafedSpread']    = bondSpread['fedSpread'].shift(1) - bondSpread['fedSpread']

bondSpread['deltaspreadBBB_lag']    = bondSpread['spreadBBB'].shift(2) - bondSpread['spreadBBB'].shift(1)
bondSpread['deltaTedSpread_lag']    = bondSpread['tedSpread'].shift(2) - bondSpread['tedSpread'].shift(1)
bondSpread['deltabbbIGSpread_lag']  = bondSpread['bbbIGSpread'].shift(2) - bondSpread['bbbIGSpread'].shift(1)
bondSpread['deltafedSpread_lag']    = bondSpread['fedSpread'].shift(2) - bondSpread['fedSpread'].shift(1)

bondSpread = bondSpread.join(pd.get_dummies(pd.rolling_mean(bondSpread['VIX'],window=3)>20,drop_first=True))
bondSpread=bondSpread.rename(columns = {True:'VIX20'})
bondSpread = bondSpread.join(pd.get_dummies(pd.rolling_mean(bondSpread['VIX'],window=3)>40,drop_first=True))
bondSpread=bondSpread.rename(columns = {True:'VIX40'})
bondSpread = bondSpread.join(pd.get_dummies(pd.to_datetime(bondSpread['Date']).dt.year.isin([2008,2009,2010]),drop_first=True))
bondSpread=bondSpread.rename(columns = {True:'Crisis'})

#Scale the Variables
scaleVars = ['deltaSpread','Slope','deltaTreas','deltaTreas3M',bondTranche+'_lag1',bondTranche+'_lag2','deltaBBB','bbbIGSpread','deltadjia','spreadBBB','deltaLibor3M','deltaLibor6M','deltaLibor1Y','tedSpread','deltaFedFunds','fedSpread','deltaSpreadOverT','deltaCREIdx','deltaspreadBBB','deltaTedSpread','deltabbbIGSpread','deltafedSpread','deltaspreadBBB_lag','deltaTedSpread_lag','deltabbbIGSpread_lag','deltafedSpread_lag']

for var in scaleVars:
    bondSpread[var] = ((bondSpread[var]*100).astype(float))

bondSpreadLag = bondSpread.dropna()
keep_vars = ['Slope','deltaSpread','deltaTreas','deltaTreas3M', bondTranche+'_lag1',bondTranche+'_lag2','VIX_lag1','VIX_lag2','deltaVIX','deltaBBB','bbbIGSpread','spreadBBB','deltadjia',bondTranche,'Crisis','VIX20','deltaLibor3M','deltaLibor6M','deltaLibor1Y','VIX40','tedSpread','deltaFedFunds','FedFunds','fedSpread','deltaSpreadOverT','deltaCREIdx','deltaspreadBBB','deltaTedSpread','deltabbbIGSpread','deltafedSpread','deltaspreadBBB_lag','deltaTedSpread_lag','deltabbbIGSpread_lag','deltafedSpread_lag','Date']

colNum =0
fig, axes = plt.subplots(figsize=(6,6), ncols=4, nrows=7)
for row in range(0,7):
    for col in range(0,4):
            if(colNum < len(keep_vars)-1):
                sns.distplot(bondSpreadLag[keep_vars[colNum]],kde=1, norm_hist=col, ax=axes[row, col])
                axes[row, col].set_title(keep_vars[colNum])
                colNum = colNum+1

colNum = 0
fig, axes = plt.subplots(figsize=(6, 6), ncols=4, nrows=7)
for row in range(0, 7):
    for col in range(0, 4):
        if (colNum < len(keep_vars)-1):
            axes[row, col].scatter(bondSpreadLag['deltaSpread'], bondSpreadLag[keep_vars[colNum]])
            axes[row, col].set_xlabel(keep_vars[colNum])
            axes[row, col].set_title(keep_vars[colNum])
            colNum = colNum + 1
            

            
#Correlation Matrix
corrMatrix = bondSpreadLag.corr()
corrMatrix['deltaSpread']
correl_vars = ['deltaSpread','Slope','deltaTreas', bondTranche+'_lag1',bondTranche+'_lag2','deltaVIX','deltaBBB','deltadjia','deltaLibor3M','tedSpread','fedSpread','deltaCREIdx']

