# For each Xi, calculate VIF
vif = pd.DataFrame([variance_inflation_factor(X.values, i) for i in range(X.shape[1])])
vif.columns = ['VIF']
vif_pretty = pd.DataFrame(explanatoryVars).join(pd.DataFrame(vif))


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

def mape(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100*np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])

def mae(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return np.mean(np.abs(ypred[idx]-ytrue[idx]))

print(mae(predActual['predSpread'], predActual['ActSpread']))
print(mae(predActualTest['predSpread'], predActualTest['ActSpread']))
print(mape(predActual['predSpread'],predActual['ActSpread']))
print(mape(predActualTest['predSpread'],predActualTest['ActSpread']))


fig = plt.figure(figsize=(12,8))
fig = smodels.graphics.plot_partregress_grid(model, fig=fig)

fig, ax = plt.subplots(figsize=(12,8))
fig = smodels.graphics.influence_plot(model, ax=ax, criterion="cooks")


from scipy import stats
res = model.resid
fig = smodels.graphics.qqplot(res, dist=stats.t, line='45', fit=True)
fig.show()


#Rsquared for predicted vs Actual
X = predActual['ActSpread'].astype(float)
y = predActual['predSpread'].astype(float)
resultActual = sm.OLS(y,X).fit()
print(resultActual.summary())

X = predActualTest['ActSpread'].astype(float)
y = predActualTest['predSpread'].astype(float)
resultActual = sm.OLS(y,X).fit()
print(resultActual.summary())

#,'Jan','Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'
## put residuals (raw & standardized) plus fitted values into a data frame
results = pd.DataFrame({
                        'resids': model.resid,
                        'std_resids': model.resid_pearson,
                        'fitted': model.predict()})

print(results.head())


# 4 plots in one window
fig = plt.figure(figsize = (8, 8), dpi = 100)

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(results['fitted'], results['resids'],  'o')
l = plt.axhline(y = 0, color = 'grey', linestyle = 'dashed')
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted')

ax2 = fig.add_subplot(2, 2, 2)
smodels.graphics.qqplot(results['std_resids'], line='s', ax = ax2)
ax2.set_title('Normal Q-Q')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(results['fitted'], abs(results['std_resids'])**.5,  'o')
ax3.set_xlabel('Fitted values')
ax3.set_ylabel('Sqrt(|standardized residuals|)')
ax3.set_title('Scale-Location')

ax4 = fig.add_subplot(2, 2, 4)
smodels.graphics.influence_plot(model, criterion = 'Cooks', size = 2, ax = ax4)

plt.tight_layout()




#Analyze Residuals
def tsplot(y, lags=None, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smodels.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax)
    smodels.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

tsplot(model.resid, lags=40)
#bondSpread1p = bondSpread.diff(periods=1, axis=0)
#bondSpread2p = bondSpread.diff(periods=2, axis=0)


fig = plt.figure("VIX")
ax = fig.add_subplot(1,1,1)
ax2 =ax.twinx()
ax.set_title("VIX", fontsize=20)
bondSpread.plot(x='Date',y='CMT10Year',ax=ax)
bondSpread.plot(x='Date',y='CMT2Year',ax=ax)

bondSpread.plot(x='Daily',y='Industrials',ax=ax)
bondSpread.plot(x='Daily',y='Financials',ax=ax)
bondSpread.plot(x='Daily',y='VIX',ax=ax2,legend = False,color='red')
