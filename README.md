# QVIP：An ILP-based Formal Verification Approach for Quantized Neural Networks
This is the official webpage for the paper QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks. In this paper, we make the following main contributions:
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

## Publications

If you use QVIP, please kindly cite our papers:
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Zhang, M., Chen, T., Sun, J.: QVIP: An ILP-based formal verification approach for quantized neural networks. In: Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022.

Some related works (verification of Quantized Neural Networks) can also be found in our other papers:
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Chen, T.: BDD4BNN: A BDD-based quantitative analysis framework for binarized neural networks. In: Proceedings of the 33rd International Conference on Computer Aided Verification, 2021.
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Chen, T.: Precise quantitative analysis of binarized neural networks: A bdd-based approach. ACM Transactions on Software Engineering and Methodology, 2023.
- Zhang, Y., Song, F., Sun, J.: QEBVerif: Quantization Error Bound Verification of Neural Networks. In: Proceedings of the 35th International Conference on Computer Aided Verification, 2023.
