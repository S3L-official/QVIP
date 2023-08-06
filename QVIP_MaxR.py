## This python can do the Maximal Robust Distance Counting with IP
## Based on gurobi
## Author: Yedi Zhang

import os
import argparse
from utils.quantization_util import *
from utils.quantized_layers import QuantizedModel
from gurobi_encoding_MaxR import *
import gurobipy as gp
from gurobipy import GRB

bigM = GRB.MAXINT

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", type=int, default=1)
parser.add_argument("--eps", type=int, default=2)
parser.add_argument("--step", type=int, default=2)
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_64")
parser.add_argument("--in_bit", type=int, default=8)
parser.add_argument("--qu_bit", type=int, default=6)
parser.add_argument("--mode", default="gp")
parser.add_argument("--outputPath", default="")

args = parser.parse_args()
print(args)
check_counter = 0

print("--eps is: " + str(args.eps))
if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)
x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)


def get_max_robust_DC_on_TF(args, lr,
                            ur):  # lr is T, ur is F, note here we count is min not robust, i.e. max robust + 1
    if ur - lr == 1:
        return lr
    else:
        mid = lr + (ur - lr) // 2
        print("lr is: ", lr)
        print("ur is: ", ur)
        print("Mid is: ", mid)
        args.eps = mid
        print(
            "***************************************************************************************************************************************************")
        print("*********************************************************** Now we check radius: ", args.eps,
              " ***********************************************************")
        print(
            "***************************************************************************************************************************************************")
        is_robust_mid = run(args)
        if is_robust_mid:
            return get_max_robust_DC_on_TF(args, mid, ur)
        else:
            return get_max_robust_DC_on_TF(args, lr, mid)


def get_interval_TF(args, lr, ur):  # first lr must be true
    print("Invoking Function get_interval_TF with lr and ur is: [ ", lr, ", ", ur, " ]")
    if ur == lr:
        ur = lr + args.step
        return get_interval_TF(args, lr, ur)

    args.eps = ur
    print("Now we check radius: ", args.eps)

    is_robust_ur = run(args)
    print("result_ur is: ", is_robust_ur)
    if is_robust_ur:  # T,T
        lr = ur
        ur = lr + args.step
        return get_interval_TF(args, lr, ur)
    else:
        return True, lr, ur


def run(args):
    global check_counter
    check_counter = check_counter + 1
    print("#################################  111  ##############################")
    arch = args.arch.split('_')
    numBlk = arch[0][:-3]
    blkset = list(map(int, arch[1:]))
    blkset.append(10)
    assert int(numBlk) == len(blkset) - 1

    weight_path = "QNN_benchmark/{}/qu_{}_{}_in_{}_qu_{}.h5".format(args.dataset, args.dataset, args.arch,
                                                                    str(args.in_bit), str(args.qu_bit))

    if blkset == [64, 10] and args.qu_bit == 6 and args.in_bit == 8:
        blkset = [64, 32] # This is for the benchmark used in AAAI21
        weight_path = "QNN_benchmark/{}_mlp.h5".format(args.dataset)

    print("weight_path is: ", weight_path)

    model = QuantizedModel(
        blkset,
        input_bits=args.in_bit,
        quantization_bits=args.qu_bit,
        last_layer_signed=True,
    )  # definition of model, print a lot of information

    print("#################################  222  ##############################")

    model.compile(  # some configurations during training
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    print("#################################  333  ##############################")

    model.build((None, 28 * 28))  # force weight allocation, print a lot of information

    print("#################################  444  ##############################")
    model.load_weights(weight_path)

    original_prediction = np.argmax(model.predict(np.expand_dims(x_test[args.sample_id], 0))[0])

    if original_prediction == y_test[args.sample_id]:
        ilp = QNNEncoding_gurobi(model, verbose=False)
        is_robust, counterexample = check_robustness_gurobi(ilp, x_test[args.sample_id].flatten(),
                                                            y_test[args.sample_id], args)
        if is_robust == True:
            print("\n{} test sample {} is robust!".format(args.dataset, args.sample_id))
        else:
            print("\n{} test sample {} is NOT robust!".format(args.dataset, args.sample_id))
    else:
        print("{} test sample {} is misclassified!".format(args.dataset, args.sample_id))
        exit(0)
    print("\nNow we finish verifying ...")
    return is_robust


original_eps = args.eps
args.eps = 1
objMax = 0
time_begin = time.time()
is_robust = run(args)
if not is_robust:
    print("\n{} test sample {} is not robust under eps {}!".format(args.dataset, args.sample_id, args.eps))
else:
    lr = 1
    ur = original_eps
    ifSucc, lb, ub = get_interval_TF(args, lr, ur)  # firs
    assert ifSucc
    objMax = get_max_robust_DC_on_TF(args, lb, ub)

time_end = time.time()
time_solving = time_end - time_begin
fo = open(args.outputPath + "/" + str(args.sample_id) + "_gp_maxR.txt", "w")
fo.write("MRR: " + str(objMax) + "\n")
fo.write("Check counter is: " + str(check_counter) + "\n")
fo.write("Check time is: " + str(time_solving))
fo.close()

print("\nMRR: " + str(objMax))
print("Check counter is: " + str(check_counter))
print("Check time is: " + str(time_solving))

