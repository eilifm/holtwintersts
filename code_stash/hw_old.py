import statsmodels.api  as sm
import pandas as pd
import numpy as np
from holtwintersts.data_generator import univ_seasonal_gen

from statsmodels.tsa.base.tsa_model import TimeSeriesModel, TimeSeriesModelResults


class HoltWinters(TimeSeriesModel):
    def __init__(self, endog, dates=None, freq=None, missing='none'):
        super(HoltWinters, self).__init__(endog, None, dates, freq, missing=missing)

        endog = self.endog  # original might not have been an ndarray

        if endog.ndim == 1:
            endog = endog[:, None]
            self.endog = endog  # to get shapes right
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")


    def fit(self, seasons=None, alpha=None, beta=None, gamma=None, auto_param=False, **kwargs):
        """
        
        Parameters
        ----------
        seasons
        alpha
        beta
        gamma
        auto_param
        kwargs

        Returns
        -------

        """

        _max_season = max(seasons)
        if _max_season > self.endog.shape[0]:
            raise ValueError("Length of data must be greater than largest season")

        # Let's init the seasonal factors and parameters
        self.seasons = seasons
        _s_factors = []

        for season in self.seasons:
             _s_factors.append(np.array([self.endog[per]/np.mean(self.endog[0:(season-1)]) for per in range(season)]))

        _L = np.mean(self.endog[0:_max_season])+np.sum([_s[-1] for _s in _s_factors])

        _B = (self.endog[_max_season] - self.endog[0])/self.endog[_max_season][0]/_max_season

        _Lm1 = _L
        _Bm1 = _B

        y_hats = np.zeros(self.endog.shape[0])

        # Iterative fit
        for t in range(self.endog.shape[0]-_max_season):
            t += _max_season

            _st_pos = np.mod(np.ones(len(seasons))*t, seasons)
            _st = np.array([_s_factors[i][int(x)] for i,x in enumerate(_st_pos)])

            _L = alpha*(self.endog[t]-np.sum(_st)) +  ((1-alpha)*(np.subtract(_L,_B)))

            _B = beta*(_L - _Lm1) + ((1-beta)*_Bm1)


            _Bm1 = _B
            _Lm1 = _L

            y_hats[t] = _L + _B + np.sum(_st)

            self.fitted = y_hats

        return HoltWintersResults(self, [_L, _B, _s_factors])


class HoltWintersResults(TimeSeriesModelResults):
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        """
        
        Class to hold results from fitting a HoltWinters model.

        Parameters
        ----------
        model : AR Model instance
            Reference to the model that is fit.
        params : array
            The fitted parameters from the AR Model.
        normalized_cov_params : array
            inv(dot(X.T,X)) where X is the lagged values.
        scale : float, optional
            An estimate of the scale of the model.
        """

        super(HoltWintersResults, self).__init__(model, params, normalized_cov_params,
                                        scale)



univ_test_data = univ_seasonal_gen([[6, 3], [12, 3]], 120, 2, scale=10)


#print([np.mean(univ_test_data.values[0::(i+1)]) for i in range(12)])


hw1 = HoltWinters(univ_test_data)
fitted_hw = hw1.fit([6, 12], .1, .5, .1)
print(fitted_hw.model.fitted)