# Harnessing Matrix Completion to Unify and Extend Viral Serology Studies
**Authors:** Tal Einav and Brian Cleary

Python and Mathematica code for low-rank matrix completion. 

---
## Python Implementation

For basic usage, a given dataset can be completed from the command line using the python implementation. For example, to complete either of the following two studies use:

###

	BASE=../
	DATA=datasets/
	OUT=results/analysis_by_random_sample/

	DATASET=CATNAP_Monoclonal_Antibodies
	python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac 1.0

	DATASET=Fonville2014_TableS1
	python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10

###

The *--obs-frac 1.0* flag indicates that all available data should be used, while *--flat-file* indicates that the first dataset is stored as a flattened array, and *--data-transform log10* in the second is used to alter the default transform, *neglog10*. The corresponding commands for each of the other included datasets are in [complete_all_datasets.sh](code/complete_all_datasets.sh).


## Basic Usage in Python

Internally, matrices are completed using the *complete_matrix* function in [matrix_completion_analysis.py](code/matrix_completion_analysis.py). The code below shows how to implement this core matrix completion algorithm based on nuclear norm minimization.

###

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

###
## Structure of Code

The [code](code/) folder reproduces the analyses in our manuscript.

We analyse antibody-virus measurements in the context of influenza (Fonville 2014) and HIV-1 (Catnap). The raw measurements are contained in the folder *datasets*. The *results* folder contains:
* Analysis by Random Sample: This approach (called intra-table completion in our manuscript) withholds a fraction of measurements within a dataset to show how well a subset of data can reconstruct the full suite of measurements. We impute these withheld values via matrix completion and quantify the accuracy of the completion. For each dataset, we show an example completions using 10%, 30%, or 50% of measurements, as well as the r^2 and RMSE curves as this fraction is varied. We also provide the completed matrix using all measurements (i.e., when no values are withheld).
* Analysis by Specific Mask: This approach (called inter-table completion in our manuscript) combines data from all six Fonville studies. For each virus, we remove all its data from one study and ask how well these data could be predicted from the other studies. In other words, can we predict how an additional virus would have looked in a dataset? We impute these values via matrix completion and quantify the accuracy of the completion.
* Analysis by Year: Using timestamps in the Catnap HIV-1 data, we use all measurements up to a certain year to predict all future measurements. In addition to the matrix-completed results, we also provide *recall curves* where the predicted titers are sorted from the strongest-to-weakest antibody interactions, and we quantify how many strong interactions could be identified using *N* experiments.

---
## Mathematica Implementation

Matrix completion is implemented via a robust PCA algorithm that takes as input a matrix whose entries are either real numbers or <code>Missing[]</code> values. The variable <code>IncompleteMatrix</code> with antibody-virus titer values is completed using:

###
	RPCA[IncompleteMatrix, "LogData" → True]
###

The option "LogData" → True indicates that completion is carried out on the Log10[values], rather than directly on the values.

### Basic Usage in Mathematica

All code and data is contained within the single file [Analysis by Mathematica - Matrix Completion.nb](code/Analysis%20by%20Mathematica%20-%20Matrix%20Completion.nb). With the Initialization section loaded, matrix completion of one influenza study would proceed as:

###
    (* Load measurements from Fonville 2014 Table S1 and matrix complete *)
    IncompleteMatrix = Table[data[{virus, serum}], {serum, sera["S1"]}, {virus, viruses["S1"]}];
    CompleteMatrix = RPCA[IncompleteMatrix, "LogData" → True];
###

Matrix completion of the HIV-1 Catnap monoclonal antibody data would proceed as:

###
    (* Load measurements of Catnap Monoclonal Antibody data and matrix complete *)
    IncompleteMatrix = Table[data[{virus, serum}], {serum, sera["Catnap"]}, {virus, viruses["Catnap"]}];
    CompleteMatrix = RPCA[IncompleteMatrix, "LogData" → True, "InvertData" → True];
###

The "InvertData" → True option specifies that the matrix values should first be inverted (value→1/value) before matrix completion. This is necessary because the most potent interactions in the pre-completion matrix should be represented by the largest values, and the Catnap monoclonal antibody data measures IC50 values where the most potent antibodies have the smallest IC50s (hence the need for inversion). Note that all other datasets examined measure serum-virus interactions in dilution units where the strongest sera are represented by the largest dilutions, and hence there is no need to invert the data.

### Structure of Code

The *Figures* section of the Mathematica notebook reproduces all plots in the manuscript. The subsection *Accessing the Data* describes how the raw measurements and the matrix-completed values can be accessed for all datasets analyzed in this work. The subsection *Methods of Matrix Completion* gives a primer on the method of matrix completion used. All parts of the notebook are heavily annotated to explain the workflow.