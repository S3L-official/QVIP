import os
import argparse
from utils.quantization_util import *
from utils.quantized_layers import QuantizedModel
from gurobi_encoding import *
import gurobipy as gp
from gurobipy import GRB

bigM = GRB.MAXINT

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", type=int, default=1)
parser.add_argument("--eps", type=int, default=2)
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--mode", default="gp")
parser.add_argument("--outputPath", default="")

args = parser.parse_args()
print(args)

print("--eps is: " + str(args.eps))
if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

y_train = y_train.flatten()  # change into one-dimension
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)  # (60000,784), before reshape: (60000,28,28)
x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)

print("#################################  111  ##############################")

model = QuantizedModel(
    [64, 32],
    input_bits=8,
    quantization_bits=6,
    last_layer_signed=True,
)  # definition of model, print a lot of information

print("#################################  222  ##############################")

weight_path = "QNN_benchmark/{}_mlp.h5".format(args.dataset)

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
    is_robust, counterexample = check_robustness_gurobi(
        ilp, x_test[args.sample_id].flatten(), y_test[args.sample_id], args.eps, args.mode, args.outputPath,
        args.sample_id
    )

    if is_robust == True:
        print("\n{} test sample {} is robust!\n".format(args.dataset, args.sample_id))
    elif is_robust == False:
        print("\n{} test sample {} is NOT robust!\n".format(args.dataset, args.sample_id))
        attacked_prediction = np.argmax(model.predict(np.expand_dims(counterexample, 0))[0])
        print(
            "Predicted original class is {}, vs attacked image (ILP) is predicted as {}".format(original_prediction,
                                                                                                attacked_prediction))
        print("predict result is: ")
        print(model.predict(np.expand_dims(counterexample, 0))[0][:10])
    else:  # is_robust is None
        print("Could not check {} test sample {}. Timeout!".format(args.dataset, args.sample_id))
else:
    print("{} test sample {} is misclassified!".format(args.dataset, args.sample_id))

print("\nNow we finish verifying ...")
