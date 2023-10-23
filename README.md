# Bilevel Optimization with a Multi-Objective Lower-Level Problem: Optimistic, Risk-Neutral, and Risk-Averse Implementations

This is the numerical implementation accompanying the paper "Bilevel Optimization with a Multi-Objective Lower-Level Problem: Risk-Neutral and Risk-Averse Formulations". Please refer to the [paper](https://arxiv.org/abs/2302.05540) for more details.

## 1. Software Requirements

The code was implemented in Python 3.9 and requires the following libraries:

+ numpy
+ torch
+ torchvision
+ sklearn
+ scipy
+ time
+ random
+ os
+ copy
+ pandas
+ seaborn
+ matplotlib
+ pickle
+ math
+ tarfile

## 2. Python Scripts

The .py files can be broken up into three categories: "solver" files, "driver" files, and "functions" files.

+ __"Solver"__ files: These files contain all of the code for the implementations of the bilevel algorithms BSG-OPT, BSG-RN, and BSG-RA.
+ __"Driver"__ files: These files contain all of the code for running the specific experiments to generate all of the plots for the six figures that are displayed in the paper.
+ __"Functions"__ files: These files contain all the the code that define the different types of problems that were tested, i.e., JOS1, SP1, and GKV1, for both the deterministic and stochastic settings.

## In case you cite our work, please refer to the paper:

T. Giovannelli, G. Kent, and L. N. Vicente. Bilevel Optimization with a Multi-Objective Lower-Level Problem: Risk-Neutral and Risk-Averse Formulations. ISE Technical Report 23T-013, Lehigh University, February 2023.


