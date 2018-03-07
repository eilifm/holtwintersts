import numpy as np
import pandas as pd
import math


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def make_seasonal_data(seasons:list, trend:int, size:int, stdev:float, start_date='2000-01-01', scale=10):
    """
    
    Parameters
    ----------
    season_lengths
    trend
    size

    Returns
    -------

    """

    seasonal_data = np.zeros(size)

    for season in range(len(seasons)):
        _sbase = np.zeros(size)+(2*math.pi/seasons[season][0])

        _sbase = np.sin(np.cumsum(_sbase))

        seasonal_data = np.add(seasonal_data, _sbase*seasons[season][1])


    return pd.DataFrame(
                np.add(
                    seasonal_data,
                    np.random.normal(0, stdev)
                    )+np.cumsum(np.zeros((size, 1))*trend/10)+scale,
                index=pd.date_range(start_date, periods=size, freq='MS')
    )


print(make_seasonal_data([[12,6], [6, 3]], 20, 120, 1.0))