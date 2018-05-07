import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from holtwintersts.data_generator import univ_seasonal_gen
from holtwintersts.hw import HoltWinters
from scipy.optimize import minimize

from sklearn.model_selection import ParameterGrid

#univ_test_data = univ_seasonal_gen([[6, 3], [12, 3]], .1, 120, .5, scale=10)
univ_test_data = univ_seasonal_gen([[7, 1], [12, 3]], .3, 120, 10, scale=15)


smallest_params = (1, .1, 1)
print(smallest_params)
best_hw = HoltWinters().fit(univ_test_data, [7, 12], smallest_params[0], smallest_params[1], smallest_params[2])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.plot(univ_test_data[96::])
ax.plot(univ_test_data)

# ax.plot(pd.Series(best_hw.predict(24), index=univ_test_data[96::].index))
ax.plot(best_hw.fitted)
# ax.plot(pd.Series(best_hw.fitted, index=univ_test_data[0:96].index))


#
plt.show()

#
# print(fitted_hw.params)
