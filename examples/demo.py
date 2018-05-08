import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from holtwintersts.data_generator import univ_seasonal_gen
from holtwintersts.hw import HoltWinters



# Generated data example
#data = univ_seasonal_gen([[6, 3], [12, 5]], .1, 120, 10, scale=15)

# Real data example
data = pd.read_csv('./ClothingSales.csv', index_col=0)

#data['Sales'] = np.add(data['Sales'], np.random.normal(0, 5000, data.shape[0]))

smallest_params = (0, 0, 0)
smallest_mse = 100000999999999999999

for alpha in np.arange(.1, .9, .1):
    for beta in np.arange(.1, .9, .1):
        for gamma in np.arange(.1, .9, .1):
            best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], alpha, beta, gamma)
            mse = mean_squared_error(data[12::], best_hw.fitted[12::])
            if mse < smallest_mse:
                smallest_mse = mse
                smallest_params = (copy.deepcopy(alpha), copy.deepcopy(beta), copy.deepcopy(gamma))
                print(smallest_mse)

            del best_hw

print(smallest_params)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(data)
best_hw = HoltWinters().fit(data, [12], *smallest_params)

ax.plot(pd.Series(best_hw.fitted, index=data.index))

plt.show()

