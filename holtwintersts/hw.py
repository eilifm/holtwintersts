import pandas as pd
import numpy as np


#from statsmodels.tsa.base.tsa_model import TimeSeriesModel, TimeSeriesModelResults


class HoltWintersResults(object):
    def __init__(self, fitted_values, endog, index, params):
        """

        Parameters
        ----------
        fitted_values
        endog
        index
        params
        """
        self.fitted = fitted_values

        self.endog = endog
        self.index = index

        if isinstance(index, pd.Index):
            self.fitted = pd.DataFrame(self.fitted, index=self.index)
            self.endog = pd.DataFrame(self.endog, index=self.index)

        for k, v in params.items():
            self.__setattr__(k, v)


    def fitted_as_dataframe(self):
        pass

    def predict(self, num_oos):
        """
        Out of sample prediction as simply a number of points from the end of the training set. This method is NOT
        index aware so, to plot, you will have to extend the dataframe index manually.

        Parameters
        ----------
        num_oos: int

        Returns
        -------
        array-like: ndarray or DataFrame


        """
        # for season in self.seasons:
        #     _s_factors.append(
        #         np.array([self.endog[per] / np.mean(self.endog[0:(season - 1)]) for per in range(season)]))

        preds = np.zeros((num_oos,))
        for samp in range(num_oos):
            for s in range(len(self.seasons)):
                # Access the correct seasonal factor
                preds[samp] += self.s_factors[s][samp % len(self.s_factors[s])]

            # Add in the level and trend components
            preds[samp] += ((self.B * samp) + self.L)

        return preds


class HoltWinters(object):
    def __init__(self, endog=None, dates=None, freq=None, missing='none'):
        """
        Holt Winters model constructor

        Parameters
        ----------
        endog
        dates
        freq
        missing
        """
        # endog = endog.copy()
        #
        # if endog.ndim == 1:
        #     endog = endog[:, None]
        #     self.endog = endog  # to get shapes right
        #     self.index = list(range(endog.shape[0]))
        #
        # elif isinstance(endog, pd.DataFrame):
        #     self.index = endog.index
        #     self.endog = endog.values
        # elif endog.ndim > 1 and endog.shape[1] != 1:
        #     raise ValueError("Only the univariate case is implemented")

    def fit(self, endog, seasons=None, alpha=None, beta=None, gamma=None, **kwargs):
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
        HoltWintersResults Object

        """
        endog = endog.copy()

        if endog.ndim == 1:
            endog = endog[:, None]
            index = list(range(endog.shape[0]))

        elif isinstance(endog, pd.DataFrame):
            index = endog.index
            endog = endog.values
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")


        _max_season = max(seasons)
        if _max_season > endog.shape[0]:
            raise ValueError("Length of data must be greater than largest season")

        # Let's init the seasonal factors and parameters
        self.seasons = seasons
        _s_factors = []
        print(_max_season)
        # Compute seasonal factors for each season
        for season in self.seasons:
            _s_factors.append(
                np.array([endog[per] / np.mean(endog[0:(season - 1)]) for per in range(season)]))


        _L = np.mean(endog[0:_max_season]) + np.sum([_s[-1] for _s in _s_factors])

        _B = (endog[_max_season] - endog[0]) / endog[_max_season][0] / _max_season

        _Lm1 = _L
        _Bm1 = _B

        y_hats = np.zeros(endog.shape[0])
        resids = np.zeros(endog.shape[0])
        # Iterative fit
        for t in range(endog.shape[0] - _max_season):
            # shift iteration to end of longest complete season
            t += _max_season

            _st_pos = np.mod(np.ones(len(seasons)) * t, seasons)

            # Get an array of the respective seasonal factors
            _st = np.array([_s_factors[i][int(x)] for i, x in enumerate(_st_pos)])

            _L = alpha * (endog[t] - np.sum(_st)) + ((1 - alpha) * (_L - _B))

            _B = (beta * (_L - _Lm1)) + ((1 - beta) * _Bm1)

            _Bm1 = _B
            _Lm1 = _L

            y_hats[t] = _L + _B + np.sum(_st)

            resids[t] = y_hats[t] - endog[t]


        params = {'alpha': alpha,
                  'beta': beta,
                  'gamma': gamma,
                  'L': _L,
                  'B': _B,
                  's_factors': _s_factors,
                  'seasons': seasons}

        print(resids)

        return HoltWintersResults(y_hats, endog, index, params)

