# imoco4e-ICTExpress2023
Code for the generation of the benchmarks used in the paper "Leveraging SMT for Verification of Neural Networks in the IMOCO4.E Project"

The benchmarks used in the paper can be found in the folder "paper_benchmarks/".

The code to replicate the generation of the benchmark is in the script benchmark_generation.py

The code to replicate the testing of the smt solvers can be found in exp_launcher.py. It should be noted that the solver
will need to be installed autonomously and minor modification to the file will be needed to replicate the experiment.

To replicate the benchmark generation the packages [pynever](https://github.com/NeVerTools/pyNeVer) (and its dependencies) and pandas are needed.