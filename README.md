# HoltWinters Time Series Forecasting

[![Build Status](https://travis-ci.org/eilifm/holtwintersts.svg?branch=master)](https://travis-ci.org/eilifm/holtwintersts)

A Python implementation of Holt-Winter's Exponential Smoothing and Forecasting in Python.

# Objective
Write a simple to read yet complete implementation of Holt Winter's smoothing/forecasting
for use by RIT students. This implementation should support multiple seasons. 

# Documentation
Unfortunately I cannot host the docs on GitHub pages at this time. To view documentation, 
download this repository and open `docs/index.html` in your browser of choice.

Docs of course can be built by `cd docs_gen && make html` from a bash shell. 

# Implementation
## Currently Implemented (original scope)
- [x] Out of sample prediction
- [x] Multiple seasons 
- [x] Additive seasonality

## Future Tasks for the Novice
- [ ] Multiplicative seasonality

## Future Tasks for the Ambitious
- [ ] Statsmodels drop-in API compatibility[1]
- [ ] Automatic DataFrame index management


## Notes
[1] The inner workings of the statsmodels API were way overkill 
for this application. The additional complexity would make this package far
too complex for a new Python user to modify. 