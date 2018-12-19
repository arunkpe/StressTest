startDate  = datetime(2008,1,1).date()
randomSampleSet = bondSpreadLag[keep_vars][bondSpreadLag.Date > startDate]
sampleDate = datetime(2015,1,1).date()
train = randomSampleSet[keep_vars][randomSampleSet.Date < sampleDate]
test = randomSampleSet[keep_vars][randomSampleSet.Date >= sampleDate]
train.reset_index(inplace=True)
test.reset_index(inplace=True)

explanatoryVars = ['deltaTreas', 'deltaTreas3M', 'VIX_lag1', 'VIX_lag2', 'deltaVIX', 'deltaBBB', 'bbbIGSpread','spreadBBB', 'deltadjia', 'Crisis', 'deltaLibor3M', 'deltaLibor6M', 'deltaLibor1Y', 'VIX40','tedSpread', 'deltaFedFunds', 'fedSpread', 'deltaSpreadOverT', 'deltaCREIdx', 'deltaspreadBBB','deltaTedSpread', 'deltabbbIGSpread', 'deltafedSpread', 'deltaspreadBBB_lag', 'deltaTedSpread_lag','deltabbbIGSpread_lag', 'deltafedSpread_lag']

import itertools as itr
import time as t
pd.options.mode.chained_assignment = None

combs = [comb for comb in itr.combinations(range(1, len(explanatoryVars)), 3)]
hi_R2 = 0
hi_mape = 150
LR_best = 0
R2_best = 0
indices_best = []
mape_train = []
mape_test = []
R2_array = []
collinearVal = []
Indices_array = []
shortlist_mape_array = []
shortlist_R2_array = []
shortlist_rms_array = []
shortlist_Indices_array = []
y = (train['deltaSpread']).astype(float)
X_temp = train[explanatoryVars]
# X = train[explanatoryVars]
X_temp = X_temp.astype(float)
X_temp_test = test[explanatoryVars]
X_temp_test = X_temp_test.astype(float)
# poly = PolynomialFeatures(degree=2)

def mape(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100*np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])

start = t.time()
for comb in combs:
    variables = X_temp.iloc[:, comb]
    if autoCorrflg == 1:
        variables = variables.join(train[autoCorr])
    variables['const'] = 1
    est_temp = smodels.OLS(y, variables)
    LR = est_temp.fit()
    R2 = LR.rsquared
    multicollinearity = LR.condition_number

    variables_test = X_temp_test.iloc[:, comb]
    if autoCorrflg == 1:
        variables_test = variables_test.join(test[autoCorr])
    variables_test['const'] = 1
    # Mape train
    bondFirstActual = bondSpread[bondSpread.Date == train['Date'][0]][bondTranche]*100
    predSpread = -1 * pd.DataFrame(LR.predict(variables))
    predSpread.iloc[0] = predSpread.iloc[0] + bondFirstActual.iloc[0]
    predSpread = predSpread.cumsum()
    predActual = (predSpread.join(pd.DataFrame(train[bondTranche]*100).astype(float)))
    predActual.columns = ['predSpread', 'ActSpread']
    mape0 = mape(predActual['predSpread'], predActual['ActSpread'])
    # Mape_Test
    bondFirstActual = bondSpread[bondSpread.Date == test['Date'][0]][bondTranche]*100
    predSpreadTest = -1 * pd.DataFrame(LR.predict(variables_test))
    predSpreadTest.iloc[0] = predSpreadTest.iloc[0] + bondFirstActual.iloc[0]
    predSpreadTest = predSpreadTest.cumsum()
    predActualTest = (predSpreadTest.join(pd.DataFrame(test[bondTranche]*100).astype(float)))
    predActualTest.columns = ['predSpread', 'ActSpread']
    mape1 = mape(predActualTest['predSpread'], predActualTest['ActSpread'])

    mape_train.append(mape0)
    mape_test.append(mape1)
    R2_array.append(R2)
    collinearVal.append(multicollinearity)
    Indices_array.append(comb)
    if (mape1 < 30) & (mape0 < 30) & (R2 > .1) & (multicollinearity < 100):
        shortlist_mape_array.append(mape1)
        shortlist_R2_array.append(R2)
        shortlist_Indices_array.append(comb)
end = t.time()

time = float((end - start) / 60)

mape_train = pd.DataFrame(mape_train)
mape_test = pd.DataFrame(mape_test)
RSq = pd.DataFrame(R2_array)
Indices = pd.DataFrame(Indices_array)
collinearVal = pd.DataFrame(collinearVal)
mape_train.columns = ['train']
mape_test.columns = ['test']
RSq.columns = ['RSquare']
collinearVal.columns = ['ConditionNum']

diagnostics = mape_train.join(mape_test).join(RSq).join(collinearVal).join(Indices)
diagnostics.dropna(inplace=True)
diagnostics.convert_objects(convert_numeric=True).sort_index(by=['test'], ascending=[True])

writer = pd.ExcelWriter('diagnosticsBondJunior.xlsx')
diagnostics.to_excel(writer, 'Sheet1')
writer.save()

explanatoryVars = ['deltaTreas', 'deltaTreas3M', 'VIX_lag1', 'VIX_lag2', 'deltaVIX', 'deltaBBB', 'bbbIGSpread','spreadBBB', 'deltadjia', 'Crisis', 'deltaLibor3M', 'deltaLibor6M', 'deltaLibor1Y', 'VIX40','tedSpread', 'deltaFedFunds', 'fedSpread', 'deltaSpreadOverT', 'deltaCREIdx', 'deltaspreadBBB','deltaTedSpread', 'deltabbbIGSpread', 'deltafedSpread', 'deltaspreadBBB_lag', 'deltaTedSpread_lag','deltabbbIGSpread_lag', 'deltafedSpread_lag']
combi = (4,22,23)
#combi = (4,8,22)

if len(combi) == 2:
    explanatoryVars = [explanatoryVars[combi[0]], explanatoryVars[combi[1]]]
if len(combi) == 3:
    explanatoryVars = [explanatoryVars[combi[0]],explanatoryVars[combi[1]],explanatoryVars[combi[2]]]
if len(combi) == 4:
    explanatoryVars = [explanatoryVars[combi[0]],explanatoryVars[combi[1]],explanatoryVars[combi[2]],explanatoryVars[combi[3]]]

y = (train['deltaSpread']).astype(float)
X = train[explanatoryVars]
if autoCorrflg == 1:
    X = X.join(train[autoCorr])
X = X.astype(float)

#est = smodels.GLM(y, X)
X = smodels.add_constant(X)
#est = smodels.GLM(y, X)
est = smodels.OLS(y, X)
model = est.fit()
model.summary()
