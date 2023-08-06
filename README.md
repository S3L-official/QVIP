# QVIP：An ILP-based Formal Verification Approach for Quantized Neural Networks
This is the official webpage for paper QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks. In this paper, we make the following main contributions:
- We propose the first ILP-based verification approach for QNNs featuring both precision and efficiency.
- We implement our approach as an end-to-end tool QVIP, using the ILP-solver Gurobi for QNN robustness verification and maximum robustness radius computation.
- We conduct an extensive evaluation of QVIP, demonstrating the efficiency and effectiveness of QVIP, e.g., significantly outperforming the state-of-the-art methods.
## Setup
Please install gurobipy and pyboolector from PyPI:
```shell script
$ pip install gurobipy
$ pip install pyboolector
```

For gurobi-solving usage, please install Gurobi on your machine.

For smt-based solving usage (AAAI21's paper: Scalable Verification of Quantized Neural Networks), please also install boolector (cf. https://github.com/mlech26l/qnn_robustness_benchmarks) 
## Running the QVIP on the benchmarks

```shell script
# Solving Mode = SMT(smt) or Gurobi(gp)

# Running QVIP on the same model used in AAAI21: 
# Network=1blk_64 (mnist), Q=6, Input=100, Attack=2, Mode=smt-solving, OutputFolder=./output
python QVIP_vs_AAAI21.py --sample_id 100 --eps 2 --dataset mnist --mode smt --outputPath output
# Network=1blk_64 (fashion-mnist), Q=6, Input=251, Attack=3, Mode=gurobi-solving, OutputFolder=./output
python QVIP_vs_AAAI21.py --sample_id 251 --eps 3 --dataset fashion-mnist --mode gp --outputPath output


# Running QVIP on the other models used in paper (ASE22-QVIP): 
# Network=1blk_100 (mnist), Q=4, Input=10, Attack=4, Mode=gurobi-solving, OutputFolder=./output
python QVIP.py --sample_id 10 --eps 4 --dataset mnist --arch 1blk_100 --qu_bit 4 --mode gp --outputPath output

# Running QVIP to compute the maximal robustness radius with the starting radius (**startR**) and **Step** size of 10
# Network=1blk_64 (mnist), Q=6, Input=698, Mode=gurobi-solving, OutputFolder=./outputMaxR
python QVIP_MaxR.py --sample_id 698 --dataset mnist --arch 1blk_64 --qu_bit 6  --eps 10 --step 10 --mode gp --outputPath outputMaxR
```
For the experimental raw data, please refer to: https://github.com/QVIP22/Data.
