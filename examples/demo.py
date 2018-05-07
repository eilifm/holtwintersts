import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from holtwintersts.data_generator import univ_seasonal_gen
from holtwintersts.hw import HoltWinters
from scipy.optimize import minimize

from sklearn.model_selection import ParameterGrid

#univ_test_data = univ_seasonal_gen([[6, 3], [12, 3]], .1, 120, .5, scale=10)
#univ_test_data = univ_seasonal_gen([[7, 1], [12, 3]], .3, 120, 10, scale=15)
data = pd.read_csv('./ClothingSales.csv', index_col=0)

data['Sales'] = np.add(data['Sales'], np.random.normal(0, 5000, data.shape[0]))

smallest_params = (.75, .5, 0)
#smallest_params = (1, 1, 1)

best_hw = HoltWinters().fit(data, [12], *smallest_params)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.plot(univ_test_data[96::])
ax.plot(data)

# ax.plot(pd.Series(best_hw.predict(24), index=univ_test_data[96::].index))
ax.plot(best_hw.fitted)
# ax.plot(pd.Series(best_hw.fitted, index=univ_test_data[0:96].index))


#
plt.show()

#
# print(fitted_hw.params)
