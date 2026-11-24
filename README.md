# MAR_MARX
Python implementation of Mixed Causal-Noncausal Autoregressive (MAR) and MAR models with exogenous variables (MARX).  Includes AML Estimation, predictive density analysis and bimodality measure for both simulations and empirical data from commodities (WTI, Natural Gas, Nickel).

File MAR_MARX_Monte_Carlo.ipynb contains the Monte Carlo simulation results for MAR(0,1), MAR(1,1) and MARX(1,1,1) with Cauchy and Student-t innovations, including for each specification: simulation of multiple processes, AML estimation, and comparison of the bimodality measure with PSY. File MAR_MARX_Simulations.ipynb contains the simulation results for one of the Monte Carlo replications, including for each model: simulation of the process, AML estimation, predictive density and bimodality measure. 

The folder Empitical Analysis contains the following files:
 - MLE_Predictive_Density.py: defines the MLE and Predictive Density functions for MAR(1,1) and MARX(1,1,1) processes under both Cauchy and Student-t innovations;
 - Functions.py: contains all relevant functions to apply AML, predictive density determination, bimodality measure and respective plots;
 - marx_bic_exhaustive.py: contains all functions for the exogenous variable selection, evaluating all possible exogenous variable subsets and selecting the one that minimizes BIC;
 - PSY.py: defines the functions used to get the PSY indicator, adapted from the psymonitor R package;
 - 3 different pickle files: save the information relative to the PSY indicator for each commodity;
 - 2 files for Natural Gas (transformed with HP-filtering an scaling by the linear trend of the volatility): one with all regressors and other with only significant regressors after variable selection has been applied to MARX Student-t case;
 - 2 files for WTI (transformed with HP-filtering an scaling by the linear trend of the volatility): one with all regressors and other with only significant regressors after variable selection has been applied to MARX Student-t case;
 - 2 files for Nickel (transformed with HP-filtering an scaling by the linear trend of the volatility): one with all regressors and other with only significant regressors after variable selection has been applied to MARX Student-t case.


The files with the data of the commodities and exogenous variables are not provided in this repository.
