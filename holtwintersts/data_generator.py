import numpy as np
import pandas as pd
import math


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def univ_seasonal_gen(seasons:list, trend:float, size:int, stdev:float, start_date='2000-01-01', scale=10):
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
    _trend_nd = np.cumsum(np.ones((size, 1))*trend)-1

    out = np.add(seasonal_data, _trend_nd)# + scale
    out = np.add(out, scale)
    out = np.add(out, np.random.normal(0, stdev, (len(seasonal_data,))))

    return pd.DataFrame(out, index=pd.date_range(start_date, periods=size, freq='MS'))


