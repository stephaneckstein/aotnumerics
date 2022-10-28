# aotnumerics

This repository supplements the paper "Computational Methods for Adapted Optimal Transport" by Stephan Eckstein and Gudmund Pammer, submitted to the Annals of Applied Probability. The repository provides code to implement the linear programming formulation for causal and bicausal optimal transport problems for time series, a backward induction implementation for the bicausal problem, and the proposed causal and bicausal versions of Sinkhorn's algorithm.

## Requirements

To run the code, we used python 3.7 and anaconda. The only non-default are scipy, scikit-learn, POT and gurobipy. Notably, for gurobipy an installed version of gurobi is necessary, which was used under an academic license.
An anaconda environment.yml file and requirements.txt file are included.
For the setup of the environment, if using anaconda, run


```setup
conda env create -f environment.yml
```

or, using pip, run

```setup
pip install -r requirements.txt
```

## Running the experiments
The values for Section 3.4 in the paper can be reproduced by running

```optimization1
python examples/ex_discretization.py
```

The saved values are provided and the Figure in Section 3.4 can be reproduced by running

```optimization1
 examples/ex_discretization_visualize.py
```

The numerics from Section 6.3. can similarly be reproduced with the programs examples/example_comparison_1.py (for Table 1) and examples/example_comparison_2.py (for Table 2), where producing the values for Table 1 from saved values can also be obtained by running examples/read_out_comparison_1.py.
