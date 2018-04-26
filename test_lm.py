import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('2012 Obesity dataset_uncorrelated.csv')
# #
# Drop NA
data = data.dropna(1)
cols = list(data.columns.values)
cols.remove("y1")
cols.remove("y2")


scaler = MinMaxScaler().fit(data.loc[:, cols])

data_s = pd.DataFrame(scaler.transform(data.loc[:, cols]), columns=cols, index=data.index)
data_s.loc[:, 'y1'] = data.loc[:, 'y1']


y, X = dmatrices('y1 ~ 1 +' + " + ".join(cols) + , data=data_s, return_type='dataframe')

model = sm.OLS(y, X).fit()
print(model.summary())

#
# print(X)
# #
# # print(data.shape)
