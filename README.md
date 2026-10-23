# MAR_MARX
Python implementation of Mixed Causal-Noncausal Autoregressive (MAR) and MAR models with exogenous variables (MARX).  Includes AML Estimation, predictive density analysis and bimodality measure for both simulations and empirical data from commodities (WTI, Natural Gas, Nickel).

File MAR_MARX_Simulations.ipynb contains the simulation results for MAR(0,1), MAR(1,1) and MARX(1,1,1) with Cauchy and Student-t innovations, including for each specification: simulation of the process, AML estimation, predictive density and bimodality measure. File MARX_Exogenous Regressors.ipynb contains simulations using different DGPs for the exogenous regressor of a MARX(1,1,1) process with Cauchy innovations.

The folder Empitical Analysis contains the following files:
 - MLE_Predictive_Density.py: defines the MLE and Predictive Density functions for MAR(1,1) and MARX(1,1,1) processes under both Cauchy and Student-t innovations;
 - Functions.py: contains all relevant functions to apply AML, predictive density determination, bimodality measure and respective plots;
 - PSY.py: defines the functions used to get the PSY indicator, adapted from the psymonitor R package;
 - 3 different pickle files: save the information relative to the PSY indicator for each commodity;
 - 6 files for Natural Gas: Natural Gas under 3 different transformations for each transformation there's a file with all regressors and other with only significant regressors (after GETS has been applied), (for the other 2 commodities only the codes of the best performing transformation are kept in this repository);
 - 2 files for WTI (transformed with HP-filtering an scaling by the linear trend of the volatility): one with all regressors and other with only significant regressors after GETS has been applied to MARX Cauchy case;
 - 2 files for Nickel (transformed with HP-filtering an scaling by the linear trend of the volatility): one with all regressors and other with only significant regressors after GETS has been applied to MARX Cauchy case.


The files with the data of the commodities and exogenous variables are not provided in this repository.
