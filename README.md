# Harnessing Matrix Completion to Unify and Extend Viral Serology Studies
Authors: Tal Einav and Brian Cleary

Python and Mathematica code for low-rank matrix completion. 

---
## Python Implementation

### Basic Usage in Python

    import numpy
    import pandas
    import cvxpy
    
    matrix_incomplete = pandas.read_csv(filename)
    # Mask for the missing data
    mask = numpy.ones(matrix_incomplete.shape,dtype=numpy.int)
	mask[numpy.where(numpy.isnan(matrix_incomplete.values))] = 0
    # Define the objective (minimize nuclear norm with an L2 penalty)
    matrix_complete = cvxpy.Variable(shape=matrix_incomplete.shape)
    objective = cvxpy.Minimize(mu * cvxpy.norm(matrix_complete, "nuc") + cvxpy.sum_squares(cvxpy.multiply(mask, matrix_complete - matrix_incomplete)))
    problem = cvxpy.Problem(objective, [])
    # Matrix completion
    problem.solve(solver=cvxpy.SCS)

### Structure of Code

The *code* folder reproduces the analyses in our manuscript. In particular, the *matrix_completion_analysis.py* file contains the matrix completion algorithm (see *complete_matrix*).

We analyse antibody-virus measurements in the context of influenza (Fonville 2014) and HIV-1 (Catnap). The raw measurements are contained in the folder *datasets*. The *results* folder contains:
* Analysis by Random Sample: This approach (called intra-table completion in our manuscript) withholds a fraction of measurements within a dataset to show how well a subset of data can reconstruct the full suite of measurements. We impute these withheld values via matrix completion and quantify the accuracy of the completion. For each dataset, we show an example completions using 10%, 30%, or 50% of measurements, as well as the r^2 and RMSE curves as this fraction is varied. We also provide the completed matrix using all measurements (i.e., when no values are withheld).
* Analysis by Specific Mask: This approach (called inter-table completion in our manuscript) combines data from all six Fonville studies. For each virus, we remove all its data from one study and ask how well these data could be predicted from the other studies. In other words, can we predict how an additional virus would have looked in a dataset? We impute these values via matrix completion and quantify the accuracy of the completion.
* Analysis by Year: Using timestamps in the Catnap HIV-1 data, we use all measurements up to a certain year to predict all future measurements. In addition to the matrix-completed results, we also provide *recall curves* where the predicted titers are sorted from the strongest-to-weakest antibody interactions, and we quantify how many strong interactions could be identified using *N* experiments.

---
## Mathematica Implementation

### Basic Usage in Mathematica

    matrixIncomplete = Import[filename]
    RPCA[matrixIncomplete, "LogData" -> True]

### Structure of Code

All code and data is contained within the file *code/Analysis By Mathematica.nb*. All plots in the manuscript were created using Mathematica and are reproduced in the *Figures* section. Matrix completion is implemented using robust PCA in the *Initialization* section.