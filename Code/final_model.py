#Final Model - re-estimated with all data points to add stability

explanatoryVars = ['deltaTreas', 'deltaTreas3M', 'VIX_lag1', 'VIX_lag2', 'deltaVIX', 'deltaBBB', 'bbbIGSpread','spreadBBB', 'deltadjia', 'Crisis', 'deltaLibor3M', 'deltaLibor6M', 'deltaLibor1Y', 'VIX40','tedSpread', 'deltaFedFunds', 'fedSpread', 'deltaSpreadOverT', 'deltaCREIdx', 'deltaspreadBBB','deltaTedSpread', 'deltabbbIGSpread', 'deltafedSpread', 'deltaspreadBBB_lag', 'deltaTedSpread_lag','deltabbbIGSpread_lag', 'deltafedSpread_lag']
if len(combi) == 2:
    explanatoryVars = [explanatoryVars[combi[0]], explanatoryVars[combi[1]]]
if len(combi) == 3:
    explanatoryVars = [explanatoryVars[combi[0]],explanatoryVars[combi[1]],explanatoryVars[combi[2]]]
if len(combi) == 4:
    explanatoryVars = [explanatoryVars[combi[0]],explanatoryVars[combi[1]],explanatoryVars[combi[2]],explanatoryVars[combi[3]]]

y = (bondSpreadLag['deltaSpread']).astype(float)
X = bondSpreadLag[explanatoryVars]
if autoCorrflg == 1:
    X = X.join(train[autoCorr])
X = X.astype(float)

X = smodels.add_constant(X)
est = smodels.OLS(y, X)
model = est.fit()
model.summary()

y = (train['deltaSpread']).astype(float)
X = train[explanatoryVars]
if autoCorrflg == 1:
    X = X.join(train[autoCorr])
X = X.astype(float)

X = smodels.add_constant(X)
bondFirstActual = bondSpread[bondSpread.Date == train['Date'][0]][bondTranche]*100
predSpread = -1*pd.DataFrame(model.predict(X))
predSpread.iloc[0] = predSpread.iloc[0]+bondFirstActual.iloc[0]
predSpread = predSpread.cumsum()
predActual = (predSpread.join(pd.DataFrame(train[bondTranche]*100).astype(float))).join(train['Date'])
predActual.columns = ['predSpread','ActSpread','Date']

fig, ax = plt.subplots()
predActual.plot(x='Date', y = 'predSpread',color ='red',ax=ax)
predActual.plot(x='Date',y='ActSpread',color = 'blue',ax=ax)

y = (test['deltaSpread']).astype(float)
X = test[explanatoryVars]
X = smodels.add_constant(X)

bondFirstActual = bondSpread[bondSpread.Date == test['Date'][0]][bondTranche]*100
predSpreadTest = -1*pd.DataFrame(model.predict(X))
predSpreadTest.iloc[0] = predSpreadTest.iloc[0]+bondFirstActual.iloc[0]
predSpreadTest = predSpreadTest.cumsum()
predActualTest = (predSpreadTest.join(pd.DataFrame(test[bondTranche]*100).astype(float))).join(test['Date'])
predActualTest.columns = ['predSpread','ActSpread','Date']

predActualTest.plot(x='Date', y = 'predSpread',color ='red',ax=ax,style = '--',legend = False)
predActualTest.plot(x='Date',y='ActSpread',color = 'blue',ax=ax,style='--',legend = False)

ax.set_ylabel('Predicted and Actual Spreads')
plt.axvspan(sampleDate, '2016-12-31', color='gray', alpha=0.2)
plt.text('2015-1-31', 500, 'Out of Sample', fontsize=12)
plt.title(bondTranche)
plt.show()
