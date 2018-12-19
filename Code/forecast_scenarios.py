
#readFile = rd.readData('FedBase.txt')
#readFile = rd.readData('FedAdv.txt')
readFile = rd.readData('FedSevAdv.txt')
macroVarName = readFile['macroVarName']
macroData    = readFile['macroData']
macroData    = macroData.replace('ND', 0)
macroData    = macroData[160:175]

bbbSpread = getMacroVar('BBB Corporate', macroVarName, geoLocation, macroData)
treas10  = getMacroVar('10 Year',macroVarName,geoLocation,macroData)
treas2   = getMacroVar('5 Year',macroVarName,geoLocation,macroData)
treas2.columns = ['Date','2 Year']
treas5   = getMacroVar('5 Year',macroVarName,geoLocation,macroData)
treas3M  = getMacroVar('3 Month Treasury',macroVarName,geoLocation,macroData)
libor1M  = getMacroVar('1 Month Libor',macroVarName,geoLocation,macroData)
libor3M  = getMacroVar('3 Month Libor',macroVarName,geoLocation,macroData)
libor6M  = getMacroVar('6 Month Libor',macroVarName,geoLocation,macroData)
fedFunds = getMacroVar('Fed Funds',macroVarName,geoLocation,macroData)
commHPI  = getMacroVar('Commercial Real Estate',macroVarName,geoLocation,macroData)
djiaIdx  = getMacroVar('Dow Jones',macroVarName,geoLocation,macroData)
vixData  = getMacroVar('Market Volatility',macroVarName,geoLocation,macroData)

treasuryCMT = treas10.set_index('Date').join(treas2.set_index('Date')).join(treas5.set_index('Date')).join(treas3M.set_index('Date')).join(libor1M.set_index('Date')).join(libor3M.set_index('Date')).join(libor6M.set_index('Date')).join(fedFunds.set_index('Date')).join(commHPI.set_index('Date')).join(bbbSpread.set_index('Date')).join(djiaIdx.set_index('Date')).join(vixData.set_index('Date'))
treasuryCMT.reset_index(inplace=True)
treasuryCMT.columns = ['Date','CMT10Year','CMT2Year','CMT5Year','CMT3M','Libor1M','Libor3M','Libor6M','FedFunds','commHPI','bbbCorp','DJIA','VIX']
treasuryCMT['Date'] = treasuryCMT['Date'].dt.date
treasuryCMT['DJIA'] = np.log(treasuryCMT['DJIA'])

treasuryCMT['Date'] = pd.to_datetime(treasuryCMT['Date'])
treasuryCMT.set_index('Date',inplace = True)
upsampled = treasuryCMT.resample('M')
treasuryCMT = upsampled.interpolate(method='cubic')
treasuryCMT.reset_index('Date',inplace = True)

bondSpreadFcst = treasuryCMT


bondSpreadFcst['Slope'] = bondSpreadFcst['CMT10Year'] - bondSpreadFcst['CMT2Year']
bondSpreadFcst['deltaTreas']      = bondSpreadFcst['CMT10Year'].shift(2) - bondSpreadFcst['CMT10Year'].shift(1)
bondSpreadFcst['VIX_lag1']        = bondSpreadFcst['VIX'].shift(2) - bondSpreadFcst['VIX'].shift(1)
bondSpreadFcst['VIX_lag2']        = bondSpreadFcst['VIX'].shift(3) - bondSpreadFcst['VIX'].shift(2)
bondSpreadFcst['deltaVIX']        = bondSpreadFcst['VIX'].shift(1) - bondSpreadFcst['VIX']
bondSpreadFcst['deltaBBB']     = bondSpreadFcst['bbbCorp'].shift(1) - bondSpreadFcst['bbbCorp']
bondSpreadFcst['spreadBBB']    = bondSpreadFcst['bbbCorp'] - bondSpreadFcst['CMT5Year']
bondSpreadFcst['deltaTreas3M'] = bondSpreadFcst['CMT3M'].shift(1)-bondSpreadFcst['CMT3M']
bondSpreadFcst['deltadjia']     = bondSpreadFcst['DJIA'].shift(2) - bondSpreadFcst['DJIA'].shift(1)
bondSpreadFcst['deltaCREIdx']     = bondSpreadFcst['commHPI'].shift(1) - bondSpreadFcst['commHPI']


bondSpreadFcst['tedSpread']    = bondSpreadFcst['Libor3M'] - bondSpreadFcst['CMT3M']
bondSpreadFcst['deltaLibor3M'] = bondSpreadFcst['Libor3M'].shift(1)-bondSpreadFcst['Libor3M']
bondSpreadFcst['deltaLibor6M'] = bondSpreadFcst['Libor6M'].shift(1)-bondSpreadFcst['Libor6M']
bondSpreadFcst['bbbIGSpread']     = bondSpreadFcst['bbbCorp'] - bondSpreadFcst['CMT10Year']
bondSpreadFcst['deltaFedFunds']     = bondSpreadFcst['FedFunds'].shift(1) - bondSpreadFcst['FedFunds']
bondSpreadFcst['fedSpread']     = (bondSpreadFcst['Libor3M'] - bondSpreadFcst['FedFunds'])

#Other Variables Created
bondSpreadFcst['deltaspreadBBB']    = bondSpreadFcst['spreadBBB'].shift(1) - bondSpreadFcst['spreadBBB']
bondSpreadFcst['deltaTedSpread']    = bondSpreadFcst['tedSpread'].shift(1) - bondSpreadFcst['tedSpread']
bondSpreadFcst['deltabbbIGSpread']  = bondSpreadFcst['bbbIGSpread'].shift(1) - bondSpreadFcst['bbbIGSpread']
bondSpreadFcst['deltafedSpread']    = bondSpreadFcst['fedSpread'].shift(1) - bondSpreadFcst['fedSpread']

bondSpreadFcst['deltaspreadBBB_lag']    = bondSpreadFcst['spreadBBB'].shift(2) - bondSpreadFcst['spreadBBB'].shift(1)
bondSpreadFcst['deltaTedSpread_lag']    = bondSpreadFcst['tedSpread'].shift(2) - bondSpreadFcst['tedSpread'].shift(1)
bondSpreadFcst['deltabbbIGSpread_lag']  = bondSpreadFcst['bbbIGSpread'].shift(2) - bondSpreadFcst['bbbIGSpread'].shift(1)
bondSpreadFcst['deltafedSpread_lag']    = bondSpreadFcst['fedSpread'].shift(2) - bondSpreadFcst['fedSpread'].shift(1)

bondSpreadFcst = bondSpreadFcst.join(pd.get_dummies(pd.rolling_mean(bondSpreadFcst['VIX'],window=3)>20,drop_first=True))
bondSpreadFcst=bondSpreadFcst.rename(columns = {True:'VIX20'})
bondSpreadFcst = bondSpreadFcst.join(pd.get_dummies(pd.rolling_mean(bondSpreadFcst['VIX'],window=3)>40,drop_first=True))
bondSpreadFcst=bondSpreadFcst.rename(columns = {True:'VIX40'})
bondSpreadFcst = bondSpreadFcst.join(pd.get_dummies(pd.to_datetime(bondSpreadFcst['Date']).dt.year.isin([2008,2009,2010]),drop_first=True))
bondSpreadFcst=bondSpreadFcst.rename(columns = {True:'Crisis'})

scaleVars = ['Slope','deltaTreas','deltaTreas3M','deltaBBB','bbbIGSpread','deltadjia','spreadBBB','deltaLibor3M','deltaLibor6M','tedSpread','deltaFedFunds','fedSpread','deltaCREIdx','deltaspreadBBB','deltaTedSpread','deltabbbIGSpread','deltafedSpread','deltaspreadBBB_lag','deltaTedSpread_lag','deltabbbIGSpread_lag','deltafedSpread_lag']
#scaleVars = ['Slope','deltaTreas','deltaTreas3M','deltaBBB','deltadjia','spreadBBB','deltaCREIdx','deltaspreadBBB','deltaspreadBBB_lag']
for var in scaleVars:
    bondSpreadFcst[var] = ((bondSpreadFcst[var]*100).astype(float))

bondSpreadLagFcst = bondSpreadFcst.dropna()
fcstDate = datetime(2017,1,31)
bondSpreadLagFcst = bondSpreadLagFcst[bondSpreadLagFcst['Date'] >= datetime(2016,12,1).date()]
bondSpreadLagFcst = bondSpreadLagFcst.reset_index()

X = bondSpreadLagFcst[explanatoryVars]
X = smodels.add_constant(X)

bondFirstActual = bondSpread[bondSpread.Date == test['Date'][24]][bondTranche]*100
predSpreadFcst = -1*pd.DataFrame(model.predict(X))
predSpreadFcst.iloc[0] = predSpreadFcst.iloc[0]+bondFirstActual.iloc[0]
predSpreadFcst = predSpreadFcst.cumsum()
predSpreadFcst = (predSpreadFcst.join(bondSpreadLagFcst['Date']))
predSpreadFcst.columns = ['Base','Date']
predSpreadFcst.columns = ['Adv','Date']
predSpreadFcst.columns = ['SevAdv','Date']

predSpreadFcst.plot(x='Date', y = 'Base',color ='black',ax=ax,legend = True)
predSpreadFcst.plot(x='Date', y = 'Adv',color ='green',ax=ax,legend = True)
predSpreadFcst.plot(x='Date', y = 'SevAdv',color ='magenta',ax=ax,legend = True)

plt.axvspan(fcstDate, '2020-12-31', color='white', alpha=0.2)
plt.text('2017-12-31', 500, 'Forecast', fontsize=12)
