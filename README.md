# Holt Winter's Time Series Forecasting

TravisCI is only testing dependencies right now. 
[![Build Status](https://travis-ci.org/eilifm/holtwintersts.svg?branch=master)](https://travis-ci.org/eilifm/holtwintersts)

A Python implementation of Holt-Winter's Exponential Smoothing and Forecasting in Python.

# Objective
Write a simple to read yet complete implementation of Holt Winter's smoothing/forecasting
for use by RIT students. This implementation should support multiple seasons. 

This was originally a school project by Eilif Mikkelsen. From time to time, contributions will continue to be made by the 
original author and other members of the community. 

# Documentation
[Documentation of Github Pages Here](http://eilif.io/holtwintersts/)

Docs of course can be built by `cd docs_gen && make html` from a bash shell. 

Be sure to commit a rebuild of the docs with a PR!

# Installation
1. Download or clone this repository.
2. Point a shell with your target Python environment to this repo's folder on your machine.
3. `pip install .` or `python setup.py install` or `python3 setup.py install`

# Examples
A basic usage example can be found in `examples/demo.py`

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
- [ ] Automatic testing suite


## Notes
[1] The inner workings of the statsmodels API were way overkill 
for this application. The additional complexity would make this package far
too complex for a new Python user to modify. 