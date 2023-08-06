from __future__ import division, print_function
import numpy as np
import pyboolector
from pyboolector import Boolector, BoolectorException
import utils.quantization_util as qu
import time
import sys
import gurobipy as gp
from gurobipy import GRB


def _renormalize(product, excessive_bits):
    shift_bits = excessive_bits
    residue = product % (2 ** shift_bits)
    c = product // (2 ** shift_bits)  # floor division
    if residue >= (2 ** (shift_bits - 1)):
        c += 1

    return np.int32(c)


def propagate_dense(in_layer, out_layer, w, b):
    for out_index in range(out_layer.layer_size):
        weight_row = w[:, out_index]
        bias_factor = (
                in_layer.frac_bits
                + in_layer.quantization_config["int_bits_bias"]
                - in_layer.quantization_config["int_bits_weights"]
        )

        bias = np.int32(b[out_index] * (2 ** bias_factor))

        bound_1 = weight_row * in_layer.clipped_lb
        bound_2 = weight_row * in_layer.clipped_ub
        accumulator_lb = np.minimum(bound_1, bound_2).sum() + bias
        accumulator_ub = np.maximum(bound_1, bound_2).sum() + bias

        accumulator_bits = (
                in_layer.bit_width + in_layer.quantization_config["quantization_bits"]
        )
        accumulator_frac = in_layer.frac_bits + (
                in_layer.quantization_config["quantization_bits"]
                - in_layer.quantization_config["int_bits_weights"]
        )

        excessive_bits = accumulator_frac - (
                in_layer.quantization_config["quantization_bits"]
                - in_layer.quantization_config["int_bits_activation"]
        )


        lb = _renormalize(accumulator_lb, excessive_bits)
        ub = _renormalize(accumulator_ub, excessive_bits)
        if out_layer.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        out_layer.accumulator_lb[out_index] = accumulator_lb
        out_layer.accumulator_ub[out_index] = accumulator_ub

        out_layer.clipped_lb[out_index] = clipped_lb
        out_layer.clipped_ub[out_index] = clipped_ub

        out_layer.lb[out_index] = lb
        out_layer.ub[out_index] = ub


class LayerEncoding_gurobi:
    def __init__(
            self,
            layer_size,
            btor,
            gp_model,
            bit_width,
            frac_bits,
            quantization_config,
            signed_output=False,
    ):
        self.layer_size = layer_size
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.quantization_config = quantization_config
        self.signed_output = signed_output

        self.vars = [
            btor.Var(btor.BitVecSort(self.bit_width)) for i in range(layer_size)
        ]

        # Gurobi TODO: assign y_variable for each output node
        if signed_output:
            self.gp_vars = [
                gp_model.addVar(lb=-50, vtype=GRB.INTEGER) for i in range(layer_size)
            ]
        else:
            self.gp_vars = [
                gp_model.addVar(vtype=GRB.INTEGER) for i in range(layer_size)
            ]
        gp_model.update()
        if self.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        self.clipped_lb = min_val * np.ones(layer_size, dtype=np.int32)
        self.clipped_ub = max_val * np.ones(layer_size, dtype=np.int32)

        acc_min, acc_max = qu.int_get_min_max_integer(30, None)
        self.accumulator_lb = acc_min * np.ones(layer_size, dtype=np.int32)
        self.accumulator_ub = acc_max * np.ones(layer_size, dtype=np.int32)

        self.lb = acc_min * np.ones(layer_size, dtype=np.int32)
        self.ub = acc_max * np.ones(layer_size, dtype=np.int32)

    # TODO: add bound constraint for gurobi
    def set_bounds(self, low, high, is_input_layer=False):
        self.lb = low
        self.ub = high

        if is_input_layer:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["input_bits"],
                self.quantization_config["input_bits"]
                - self.quantization_config["int_bits_input"],
            )
        elif self.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        self.clipped_lb = np.clip(self.lb, min_val, max_val)
        self.clipped_ub = np.clip(self.ub, min_val, max_val)


class QNNEncoding_gurobi:
    def __init__(self, quantized_model, btor=None, verbose=None, config=None):

        if btor is None:
            btor = Boolector()
            btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 2)
        self.btor = btor
        self.gp_model = gp.Model("qnn_ilp_verifier")
        self.gp_model.Params.IntFeasTol = 1e-9
        self.gp_model.setParam(GRB.Param.Threads, 28)
        self._debug_list = []
        self._verbose = verbose
        self.config = {
            "add_bound_constraints": True,
            "sat_engine_cadical": True,
            "rebalance_sum": True,
            "recursive_sum": True,
            "sort_sum": None,
            "propagate_bounds": True,
            "relu_simplify": True,
            "subsum_elimination": False,  # subsub elimination can hurt performance -> better not activate it
            "min_bits": True,
            "shift_elimination": True,
        }
        # Overwrite default config with argument
        if not config is None:
            for k, v in config.items():
                self.config[k] = v

        self._stats = {
            "constant_neurons": 0,
            "linear_neurons": 0,
            "unstable_neurons": 0,
            "reused_expressions": 0,
            "partially_stable_neurons": 0,
            "build_time": 0,
            "smt_sat_time": 0,
            "gp_sat_time": 0,
            "total_time": 0,
        }

        self.dense_layers = []
        self.quantized_model = quantized_model

        self._last_layer_signed = quantized_model._last_layer_signed  # True
        self.quantization_config = quantized_model.quantization_config

        current_bits = self.quantization_config["input_bits"]  # 8 bits
        for i, l in enumerate(quantized_model.dense_layers):
            self.dense_layers.append(
                LayerEncoding_gurobi(
                    layer_size=l.units,
                    btor=self.btor,
                    gp_model=self.gp_model,
                    bit_width=self.quantization_config["quantization_bits"],
                    frac_bits=self.quantization_config["quantization_bits"]
                              - self.quantization_config["int_bits_activation"],
                    quantization_config=self.quantization_config,
                    signed_output=l.signed_output,
                )
            )
        # Create input vars
        input_size = quantized_model._input_shape[-1]
        self.input_layer = LayerEncoding_gurobi(
            layer_size=input_size,
            btor=self.btor,
            gp_model=self.gp_model,
            bit_width=self.quantization_config["input_bits"],
            frac_bits=self.quantization_config["input_bits"]
                      - self.quantization_config["int_bits_input"],
            quantization_config=self.quantization_config,
        )
        self.input_vars = self.input_layer.vars
        self.output_vars = self.dense_layers[-1].vars

        self.input_gp_vars = self.input_layer.gp_vars
        self.output_gp_vars = self.dense_layers[-1].gp_vars

    def print_verbose(self, text):
        if self._verbose:
            print(text)

    def only_encode(self):
        if self.config["propagate_bounds"]:
            self.propagate_bounds()
        self.encode()

    def sat(self, args, timeout=None, verbose=False):
        build_start_time = time.time()
        if self.config["propagate_bounds"]:
            if verbose:
                print("Propagating bounds ...")
            self.propagate_bounds()

        if verbose:
            print("Encode model ...")
        self.encode()

        if not timeout is None:
            self.btor.Set_term(lambda x: time.time() - x > timeout, time.time())

        if self.config["sat_engine_cadical"]:
            self.btor.Set_sat_solver("cadical")
        else:
            self.btor.Set_sat_solver("lingeling")

        if verbose:
            print("Invoking SMT engine ...")

        if args.mode == "gp":

            gp_solving_start_time = time.time()
            self._stats["build_time"] = gp_solving_start_time - build_start_time
            print("\n The encoding time is: " + str(self._stats["build_time"]))
            print(
                "\n==================================== Now we start do ilp-based solving! ====================================")

            self.gp_model.optimize()
            gp_end_time = time.time()
            self._stats["gp_sat_time"] = gp_end_time - gp_solving_start_time

            print("===== ===== GP sat time is: " + str(self._stats["gp_sat_time"]) + " ===== =====\n")
            ifgpSat = self.gp_model.status == 2
            print("ifgpSat: " + str(ifgpSat))
            fo = open(args.outputPath + "/" + str(args.sample_id) + "_gp.txt", "w")
            fo.write("Verification Result: " + str(ifgpSat) + "\n")
            fo.write("Encoding Time: " + str(self._stats["build_time"]) + "\n")
            fo.write("Solving Time: " + str(self._stats["gp_sat_time"]))
            return ifgpSat

        else:
            print("Wrong mode type")
            exit(0)

    def propagate_bounds(self):

        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            self.print_verbose("propagate Dense layer")
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            propagate_dense(current_layer, l, w, b)

            current_layer = l

    def encode(self):

        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            self.print_verbose("encoding Dense layer")
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            self.encode_dense(current_layer, l, w, b)

            current_layer = l

    def get_accumulator_bits_layerwide(self, in_layer, out_layer, b):
        return 1 + int(
            np.max(
                [
                    self.get_accumulator_bits_individually(in_layer, out_layer, b, i)
                    for i in range(out_layer.layer_size)
                ]
            )
        )

    def get_accumulator_bits_individually(self, in_layer, out_layer, b, out_index):
        bias_factor = (
                in_layer.frac_bits
                + self.quantization_config["int_bits_bias"]
                - self.quantization_config["int_bits_weights"]
        )
        bias = int(b[out_index] * (2 ** bias_factor))

        abs_max_value = 1 + np.max(
            [
                np.abs(out_layer.accumulator_lb[out_index]),
                np.abs(out_layer.accumulator_ub[out_index]),
            ]
        )

        accumulator_bits = np.max(
            [
                int(np.ceil(np.log2(abs_max_value))) + 1,
                int(np.ceil(np.log2(np.abs(bias) + 1))) + 1,
                8,
            ]
        )
        return 1 + accumulator_bits

    def encode_dense(self, in_layer, out_layer, w, b):

        self.clear_scratchpad()
        if self.config["subsum_elimination"]:  # not jump in here
            if self.config["min_bits"]:
                accumulator_bits = 1 + self.get_accumulator_bits_layerwide(
                    in_layer, out_layer, b
                )
            else:
                accumulator_bits = 32
            sign_extended_x = [
                self.btor.Uext(in_layer.vars[i], accumulator_bits - in_layer.bit_width)
                for i in range(in_layer.layer_size)
            ]
            self.subexpression_analysis(in_layer, sign_extended_x, w)
        for out_index in range(out_layer.layer_size):
            weight_row = w[:, out_index]
            bias_factor = (
                    in_layer.frac_bits
                    + self.quantization_config["int_bits_bias"]
                    - self.quantization_config["int_bits_weights"]
            )
            bias = int(b[out_index] * (2 ** bias_factor))

            if not self.config["subsum_elimination"]:
                if self.config["min_bits"]:
                    accumulator_bits = 1 + self.get_accumulator_bits_individually(
                        in_layer, out_layer, b, out_index
                    )
                else:
                    accumulator_bits = 32
                sign_extended_x = [
                    self.btor.Uext(
                        in_layer.vars[i], accumulator_bits - in_layer.bit_width
                    )
                    for i in range(in_layer.layer_size)
                ]
            if not self.config["min_bits"]:
                accumulator_bits = 32

            id_var_weight_list = [
                (i, sign_extended_x[i], int(weight_row[i]))
                for i in range(len(sign_extended_x))
            ]

            gp_id_var_weight_list = [
                (i, in_layer.gp_vars[i], int(weight_row[i]))
                for i in range(len(in_layer.gp_vars))
            ]

            if self.config["rebalance_sum"]:  # jump in here
                id_var_weight_list = self.prune_zeros(id_var_weight_list)
                gp_id_var_weight_list = self.prune_zeros(gp_id_var_weight_list)
            if not self.config["sort_sum"] is None:  # not jump in here
                id_var_weight_list = self.sort_sum(
                    id_var_weight_list,
                    ascending=self.config["sort_sum"].startswith("asc"),
                )
            accumulator = self.reduce_MAC(
                id_var_weight_list, subsum_elimination=self.config["subsum_elimination"]
            )

            gp_accumulator = self.gp_reduce_MAC(
                gp_id_var_weight_list, subsum_elimination=self.config["subsum_elimination"]
            )

            if accumulator is None:
                exit(0)
                # Neuron is constant 0
                accumulator = self.btor.Const(bias, accumulator_bits)
                gp_accumulator = bias
            else:
                accumulator = self.btor.Add(accumulator, bias)
                gp_accumulator = gp_accumulator + bias

            accumulator_bits = (
                    in_layer.bit_width + self.quantization_config["quantization_bits"]
            )
            accumulator_frac = in_layer.frac_bits + (
                    self.quantization_config["quantization_bits"]
                    - self.quantization_config["int_bits_weights"]
            )

            excessive_bits = accumulator_frac - (
                    self.quantization_config["quantization_bits"]
                    - self.quantization_config["int_bits_activation"]
            )

            self.renormalize(
                value=accumulator,
                lb=out_layer.lb[out_index],
                ub=out_layer.ub[out_index],
                output_var=out_layer.vars[out_index],
                signed_output=out_layer.signed_output,
                excessive_bits=excessive_bits,
                aux="dense {} out_index: {}".format(out_layer.signed_output, out_index),
            )

            # TODO: recheck-here!!!!
            self.gp_renormalize(
                value=accumulator,
                gp_value=gp_accumulator,
                lb=out_layer.lb[out_index],
                ub=out_layer.ub[out_index],
                output_var=out_layer.vars[out_index],
                gp_output_var=out_layer.gp_vars[out_index],
                signed_output=out_layer.signed_output,
                excessive_bits=excessive_bits,
                aux="dense {} out_index: {}".format(out_layer.signed_output, out_index),
            )

    def prune_zeros(self, id_var_weight_list):
        new_list = []
        for i, x, w in id_var_weight_list:
            if w != 0:
                new_list.append((i, x, w))
        return new_list

    def sort_sum(self, id_var_weight_list, ascending=True):

        factor = 1 if ascending else -1
        id_var_weight_list.sort(key=lambda tup: factor * int(np.abs(tup[-1])))

        return id_var_weight_list

    def clear_scratchpad(self):
        self._scatchpad = {}

    def _add_expr_to_scratchpad(self, in_index, w_value, expr):
        if not in_index in self._scatchpad.keys():
            self._scatchpad[in_index] = {}
        self._scatchpad[in_index][w_value] = expr

    def _get_expr_from_scratchpad(self, in_index, w_value):
        try:
            return self._scatchpad[in_index][w_value]
        except:
            s = "None"
            if in_index in self._scatchpad.keys():
                s = str(self._scatchpad[in_index])
            print("\nQuery: [{}][{}]".format(in_index, w_value))
            print(
                "Screthpad key error! keys: {}, item: {}".format(
                    self._scatchpad.keys(), s
                )
            )
            print("ERROR")
            import sys

            sys.exit(-1)

    def _build_pos_expression(self, in_index, v, all_pos_values):
        sext_in_var = self._get_expr_from_scratchpad(in_index, 1)
        expr = None
        assert not sext_in_var is None
        if v == 1:
            return sext_in_var

        if self.config["shift_elimination"]:
            for shift in [1, 2, 3, 4, 5, 6]:
                mul_val = 2 ** shift
                if (
                        v > mul_val
                        and all_pos_values[v // mul_val] > 0
                        and v % mul_val == 0
                ):
                    expr = self._get_expr_from_scratchpad(in_index, v // mul_val)
                    expr = self.btor.Mul(expr, mul_val)
                    return expr

        expr = self.btor.Mul(sext_in_var, v)
        return expr

    def _build_neg_expression(self, in_index, v, all_pos_values, all_neg_values):
        if all_pos_values[v] > 0:
            sext_in_var = self._get_expr_from_scratchpad(in_index, v)
            assert not sext_in_var is None
            return self.btor.Neg(sext_in_var)

        if self.config["shift_elimination"]:
            for shift in [1, 2, 3, 4, 5, 6]:
                mul_val = 2 ** shift
                if (
                        v > mul_val
                        and all_neg_values[v // mul_val] > 0
                        and v % mul_val == 0
                ):
                    expr = self._get_expr_from_scratchpad(in_index, -v // mul_val)
                    expr = self.btor.Mul(expr, mul_val)
                    return expr

        sext_in_var = self._get_expr_from_scratchpad(in_index, 1)
        assert not sext_in_var is None
        expr = self.btor.Mul(sext_in_var, -v)
        return expr

    def subexpression_analysis(self, in_layer, sext_in_vars, weight_matrix):
        for in_index in range(weight_matrix.shape[0]):
            all_pos_values = np.zeros(
                2 ** (self.quantization_config["quantization_bits"] - 1) + 1,
                dtype=np.int32,
            )
            all_neg_values = np.zeros(
                2 ** (self.quantization_config["quantization_bits"] - 1) + 1,
                dtype=np.int32,
            )
            for o in range(weight_matrix.shape[1]):
                w = int(weight_matrix[in_index, o])
                if w >= 0:
                    all_pos_values[w] += 1
                else:
                    all_neg_values[-w] += 1

            self._add_expr_to_scratchpad(in_index, 1, sext_in_vars[in_index])
            for v in range(
                    1, 2 ** (self.quantization_config["quantization_bits"] - 1) + 1
            ):
                if all_pos_values[v] > 0:
                    expr = self._build_pos_expression(in_index, v, all_pos_values)
                    self._add_expr_to_scratchpad(in_index, v, expr)
                if all_neg_values[v] > 0:
                    expr = self._build_neg_expression(
                        in_index, v, all_pos_values, all_neg_values
                    )
                    self._add_expr_to_scratchpad(in_index, -v, expr)

    def reduce_MAC(self, id_var_weight_list, subsum_elimination):
        if len(id_var_weight_list) == 1:
            i, x, weight_value = id_var_weight_list[0]
            if weight_value == 0:
                return None
            if subsum_elimination:
                expr = self._get_expr_from_scratchpad(i, weight_value)
                if expr is None:
                    raise ValueError(
                        "Missing expression x[{}]*{}".format(i, weight_value)
                    )
                if x.width - expr.width > 0:
                    expr = self.btor.Sext(expr, x.width - expr.width)
                    return expr
                    print(
                        "WARNING: Sub-sum has more bits than total sum ({} and {}) falling back to standard multiplication".format(
                            str(x.width), str(expr.width)
                        )
                    )

            if weight_value == 1:
                return x
            elif weight_value == -1:
                return self.btor.Neg(x)
            else:
                return self.btor.Mul(x, weight_value)
        else:
            if self.config["recursive_sum"]:
                center = len(id_var_weight_list) // 2
            else:
                center = 1
            left = self.reduce_MAC(id_var_weight_list[:center], subsum_elimination)
            right = self.reduce_MAC(id_var_weight_list[center:], subsum_elimination)
            if left is None:
                return right
            elif right is None:
                return left
            return self.btor.Add(left, right)

    def gp_reduce_MAC(self, id_var_weight_list, subsum_elimination):
        if len(id_var_weight_list) == 1:
            i, x, weight_value = id_var_weight_list[0]
            if weight_value == 0:
                return None

            if weight_value == 1:
                return x
            elif weight_value == -1:
                return -x
            else:
                return x * weight_value
        else:
            if self.config["recursive_sum"]:
                center = len(id_var_weight_list) // 2
            else:
                center = 1
            left = self.gp_reduce_MAC(id_var_weight_list[:center], subsum_elimination)
            right = self.gp_reduce_MAC(id_var_weight_list[center:], subsum_elimination)
            if left is None:
                return right
            elif right is None:
                return left
            return left + right

    def renormalize(
            self, value, lb, ub, output_var, signed_output, excessive_bits, aux=None
    ):
        assert lb <= ub
        if signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        if self.config["relu_simplify"] and clipped_ub == clipped_lb:
            self.btor.Assert(self.btor.Eq(output_var, int(clipped_ub)))
            self.print_verbose("neuron fixed at {}".format(int(clipped_ub)))
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_lb >= max_val:
            exit(0)
            self.print_verbose("neuron fixed at ub {}".format(max_val))
            self.btor.Assert(self.btor.Eq(output_var, max_val))
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_ub <= min_val:
            exit(0)
            self.btor.Assert(self.btor.Eq(output_var, min_val))
            self.print_verbose("neuron fixed at lb {}".format(min_val))
            self._stats["constant_neurons"] += 1
            return

        residue = self.btor.Slice(value, excessive_bits - 1, 0)
        quotient = self.btor.Slice(
            value, value.width - 1, excessive_bits
        )

        rouned_output = self.btor.Var(
            self.btor.BitVecSort(value.width - excessive_bits)
        )

        residue_threshold = 2 ** (excessive_bits - 1)
        self.btor.Assert(
            (
                    self.btor.Ugte(residue, residue_threshold)
                    & self.btor.Eq(rouned_output, self.btor.Inc(quotient))
            )
            | (
                    self.btor.Not(self.btor.Ugte(residue, residue_threshold))
                    & self.btor.Eq(rouned_output, quotient)
            )
        )

        sign_ext_func = self.btor.Uext
        gte_func = self.btor.Ugte
        lte_func = self.btor.Ulte
        gt_func = self.btor.Ugt
        lt_func = self.btor.Ult
        if signed_output:
            gte_func = self.btor.Sgte
            lte_func = self.btor.Slte
            gt_func = self.btor.Sgt
            lt_func = self.btor.Slt
            sign_ext_func = self.btor.Sext
        if rouned_output.width - output_var.width < 0:
            rouned_output = self.btor.Sext(
                rouned_output, -(rouned_output.width - output_var.width)
            )
        output_var_sign_ext = sign_ext_func(
            output_var, rouned_output.width - output_var.width
        )

        if (self.config["relu_simplify"]) and lb >= min_val and ub <= max_val:
            self.btor.Assert(self.btor.Eq(output_var_sign_ext, rouned_output))
            self.print_verbose(
                "neuron is linear: [{}, {}] clippling range: [{}, {}]".format(
                    lb, ub, min_val, max_val
                )
            )
            self._stats["linear_neurons"] += 1
        elif self.config["relu_simplify"] and lb >= min_val:
            # low clipping is impossible
            self.btor.Assert(
                (
                        self.btor.Sgt(rouned_output, max_val)
                        & self.btor.Eq(output_var, max_val)
                )
                | (
                        self.btor.Slte(rouned_output, max_val)
                        & self.btor.Eq(rouned_output, output_var_sign_ext)
                )
            )
            self.print_verbose("neuron cannot reach min_val of {}".format(min_val))
            self._stats["partially_stable_neurons"] += 1
        elif self.config["relu_simplify"] and ub <= max_val:
            # high clipping is impossible
            self.btor.Assert(
                (
                        self.btor.Slt(rouned_output, min_val)
                        & self.btor.Eq(output_var, min_val)
                )
                | (
                        self.btor.Sgte(rouned_output, min_val)
                        & self.btor.Eq(rouned_output, output_var_sign_ext)
                )
            )
            self.print_verbose("neuron cannot reach max val of {}".format(max_val))
            self._stats["partially_stable_neurons"] += 1
        else:
            # Don't know anything about the neuron
            self.btor.Assert(
                (
                        self.btor.Slt(rouned_output, min_val)
                        & self.btor.Eq(output_var, min_val)
                )
                | (
                        self.btor.Sgt(rouned_output, max_val)
                        & self.btor.Eq(output_var, max_val)
                )
                | (
                        self.btor.Sgte(rouned_output, min_val)
                        & self.btor.Slte(rouned_output, max_val)
                        & self.btor.Eq(output_var_sign_ext, rouned_output)
                )
            )
            # self.print_verbose("dont't know anything about neuron")
            self._stats["unstable_neurons"] += 1

        # Add bound constraints, output_var (4 bit) is unsigned!!!!
        if self.config["add_bound_constraints"]:
            if clipped_lb > min_val:
                self.btor.Assert(gte_func(output_var, int(clipped_lb)))
            if clipped_ub < max_val:
                self.btor.Assert(lte_func(output_var, int(clipped_ub)))
            # self._debug_list.append((output_var,int(clipped_lb),int(clipped_ub),aux))

    # TODO: rewrite renormalize function for gurobi
    def gp_renormalize(
            self, value, gp_value, lb, ub, output_var, gp_output_var, signed_output, excessive_bits, aux=None
    ):
        assert lb <= ub
        if signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        if self.config["relu_simplify"] and clipped_ub == clipped_lb:
            self.print_verbose("neuron fixed at {}".format(int(clipped_ub)))
            self.gp_model.addConstr(gp_output_var == int(clipped_ub))
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_lb >= max_val:  # 无用
            exit(0)
            print("max_val is: " + str(max_val))
            self.print_verbose("neuron fixed at ub {}".format(max_val))
            self.gp_model.addConstr(gp_output_var == max_val)
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_ub <= min_val:
            exit(0)
            self.gp_model.addConstr(gp_output_var, min_val)
            print("min_val is: " + str(min_val))
            self.print_verbose("neuron fixed at lb {}".format(min_val))
            self._stats["constant_neurons"] += 1
            return

        gp_lb = max(lb, min_val)
        gp_ub = min(ub, max_val)

        band = gp_ub - gp_lb + 1
        aux_y_vars = [self.gp_model.addVar(vtype=GRB.BINARY) for i in range(band)]
        self.gp_model.update()

        tmp_output_var = 0
        tmp_aux_y = 0
        tmp_x_constr1_ub = 0
        tmp_x_constr2_lb = 0

        for i in range(band):
            tmp_output_var = tmp_output_var + int(gp_lb + i) * aux_y_vars[i]
            tmp_aux_y = tmp_aux_y + aux_y_vars[i]
            if i < band - 1:
                tmp_x_constr1_ub = tmp_x_constr1_ub + (gp_lb + 0.5 + i) * aux_y_vars[i]
                tmp_x_constr2_lb = tmp_x_constr2_lb + (gp_lb + 0.5 + i) * aux_y_vars[i + 1]

        tmp_x_constr1_ub = tmp_x_constr1_ub + GRB.MAXINT * aux_y_vars[band - 1]
        tmp_x_constr2_lb = tmp_x_constr2_lb - GRB.MAXINT * aux_y_vars[0]

        self.gp_model.addConstr(tmp_aux_y == 1)
        self.gp_model.addConstr(gp_output_var == tmp_output_var)
        gp_quotient = gp_value / (2 ** excessive_bits)  # x
        self.gp_model.addConstr(gp_quotient <= tmp_x_constr1_ub - 0.0001)  # 10:0.0009 9: 0.001953125 5:0.03125
        self.gp_model.addConstr(gp_quotient >= tmp_x_constr2_lb)

        if self.config["add_bound_constraints"]:
            if clipped_lb > min_val:
                self.gp_model.addConstr(gp_output_var >= int(clipped_lb))
            if clipped_ub < max_val:
                self.gp_model.addConstr(gp_output_var <= int(clipped_ub))

    def print_stats(self):
        print("Constant neurons: {:d}".format(self._stats["constant_neurons"]))
        print("Linear   neurons: {:d}".format(self._stats["linear_neurons"]))
        print("Bistable neurons: {:d}".format(self._stats["partially_stable_neurons"]))
        print(
            "Unstable neurons: {:d} (tri-stable)".format(self._stats["unstable_neuron"])
        )

    def assert_input_box(self, low, high, set_bounds=True):

        input_size = len(self.input_vars)

        low = np.array(low, dtype=np.int32) * np.ones(input_size, dtype=np.int32)
        high = np.array(high, dtype=np.int32) * np.ones(input_size, dtype=np.int32)

        saturation_min, saturation_max = qu.uint_get_min_max_integer(
            self.quantization_config["input_bits"],
            self.quantization_config["input_bits"]
            - self.quantization_config["int_bits_input"],
        )

        low = np.clip(low, saturation_min, saturation_max)
        high = np.clip(high, saturation_min, saturation_max)
        if set_bounds:
            self.input_layer.set_bounds(low, high, is_input_layer=True)

        for i in range(input_size):
            self.btor.Assert(self.btor.Ugte(self.input_layer.vars[i], int(low[i])))
            self.btor.Assert(self.btor.Ulte(self.input_layer.vars[i], int(high[i])))

        for i in range(input_size):
            self.gp_model.addConstr(self.input_layer.gp_vars[i] >= int(low[i]))
            self.gp_model.addConstr(self.input_layer.gp_vars[i] <= int(high[i]))

    def sanitize(self):
        for var, lb, ub, aux in self._debug_list:
            value = qu.binary_str_to_uint(var.assignment)
            if value < lb or value > ub:
                print(
                    "ERROR: value: {}, range [{}, {}] aux: {}".format(
                        value, lb, ub, str(aux)
                    )
                )

    def assert_not_argmax(self, max_index):
        print("max_index is: " + str(max_index))
        lte_func = self.btor.Slte if self._last_layer_signed else self.btor.Ulte
        lt_func = self.btor.Slt if self._last_layer_signed else self.btor.Ult

        disjunction = None
        for i in range(len(self.output_vars)):
            if i == int(max_index):
                continue
            if i < int(max_index):
                or_term = lte_func(
                    self.output_vars[int(max_index)], self.output_vars[i]
                )
            else:
                or_term = lt_func(self.output_vars[int(max_index)], self.output_vars[i])
            if disjunction is None:
                disjunction = or_term
            else:
                disjunction = self.btor.Or(disjunction, or_term)

        self.btor.Assert(disjunction)

    def assert_not_argmax_test(self, max_index):
        lte_func = self.btor.Slte if self._last_layer_signed else self.btor.Ulte
        tmp_pre = 0
        tmp_pos = 0
        tmp_smt_pre = 0
        tmp_smt_pos = 0
        for i in range(5):
            tmp_pre = tmp_pre + self.output_gp_vars[i]
            var_i = self.btor.Sext(self.output_vars[i], 4)
            tmp_smt_pre = tmp_smt_pre + var_i

        for i in range(5):
            tmp_pos = tmp_pos + self.output_gp_vars[i + 5]
            var_i = self.btor.Sext(self.output_vars[i + 5], 4)
            tmp_smt_pos = tmp_smt_pos + var_i

        self.gp_model.addConstr(tmp_pre <= tmp_pos)  # sat
        disjunction = lte_func(tmp_smt_pre, tmp_smt_pos)  # sat
        self.btor.Assert(disjunction)

    def assert_not_argmax_gurobi(self, max_index):
        bigM = GRB.MAXINT
        k_list = []
        for i in range(len(self.output_vars)):
            # for i in range(10):
            if i == int(max_index):
                continue
            if i < int(max_index):
                k_i = self.gp_model.addVar(vtype=GRB.BINARY)
                k_list.append(k_i)
                self.gp_model.update()
                self.gp_model.addConstr(
                    self.output_gp_vars[i] >= self.output_gp_vars[int(max_index)] + bigM * (k_i - 1))
                self.gp_model.addConstr(
                    self.output_gp_vars[i] <= self.output_gp_vars[int(max_index)] + bigM * k_i - 1)
            else:
                k_i = self.gp_model.addVar(vtype=GRB.BINARY)
                k_list.append(k_i)
                self.gp_model.update()
                self.gp_model.addConstr(
                    self.output_gp_vars[i] >= self.output_gp_vars[int(max_index)] + 1 + bigM * (k_i - 1))
                self.gp_model.addConstr(
                    self.output_gp_vars[i] <= self.output_gp_vars[int(max_index)] + bigM * k_i)

        print("k_list is: ")
        print(k_list)
        sum_constr = 0
        self.gp_model.update()

        for i in range(len(k_list)):
            sum_constr = sum_constr + k_list[i]

        self.gp_model.addConstr(sum_constr >= 1)

    def assert_argmax(self, var_list, max_index):
        gte_func = self.btor.Sgte if self._last_layer_signed else self.btor.Ugte
        gt_func = self.btor.Sgt if self._last_layer_signed else self.btor.Ugt

        for i in range(len(var_list)):
            if i == int(max_index):
                continue
            if i < int(max_index):
                self.btor.Assert(gte_func(var_list[int(max_index)], var_list[i]))
            else:
                self.btor.Assert(gt_func(var_list[int(max_index)], var_list[i]))

    def assert_input_distance(self, var_list, distance):
        for i in range(len(var_list)):
            greater = self.btor.Ugte(self.input_vars[i], var_list[i])
            self.btor.Assert(
                self.btor.Implies(
                    greater,
                    self.btor.Ulte(
                        self.btor.Sub(self.input_vars[i], var_list[i]), distance
                    ),
                )
            )
            self.btor.Assert(
                self.btor.Implies(
                    self.btor.Not(greater),
                    self.btor.Ulte(
                        self.btor.Sub(var_list[i], self.input_vars[i]), distance
                    ),
                )
            )

    def assert_not_output_distance(self, var_list, distance):
        self._output_dbg_expr = []
        for i in range(len(var_list)):
            if self._last_layer_signed:
                v1 = self.btor.Sext(self.output_vars[i], 1)
                v2 = self.btor.Sext(var_list[i], 1)
            else:
                v1 = self.btor.Uext(self.output_vars[i], 1)
                v2 = self.btor.Uext(var_list[i], 1)
            sub = self.btor.Sub(v1, v2)
            expr = self.btor.Sgte(sub, int(distance)) | self.btor.Slte(
                sub, -int(distance)
            )
            self.btor.Assert(expr)
            self._output_dbg_expr.append(expr)

    def get_input_assignment(self):
        input_values = np.array(
            [qu.binary_str_to_uint(var.assignment) for var in self.input_layer.vars]
        )
        return np.array(input_values, dtype=np.int32)

    def get_input_assignment_gurobi(self):
        input_values_gp = np.array(
            [var.X for var in self.input_layer.gp_vars]
        )
        return np.array(input_values_gp, dtype=np.int32)

    def get_forward_buffer(self):
        print("_last_layer_signed: ", str(self._last_layer_signed))
        input_values = np.array(
            [qu.binary_str_to_uint(var.assignment) for var in self.input_layer.vars],
            dtype=np.int32,
        )
        activation_buffer = []
        for i in range(len(self.dense_layers) - 1):
            values = np.array(
                [
                    qu.binary_str_to_uint(var.assignment)
                    for var in self.dense_layers[i].vars
                ],
                dtype=np.int32,
            )
            activation_buffer.append(values)
        decode_func = (
            qu.binary_str_to_int if self._last_layer_signed else qu.binary_str_to_uint
        )
        output_values = np.array(
            [decode_func(var.assignment) for var in self.output_vars], dtype=np.int32
        )
        activation_buffer.append(output_values)
        return activation_buffer

    def get_output_assignment(self):
        output_values = np.empty(len(self.output_vars), dtype=np.int32)
        decode_func = (
            qu.binary_str_to_int if self._last_layer_signed else qu.binary_str_to_uint
        )
        for i in range(len(self.output_vars)):
            output_values[i] = decode_func(self.output_vars[i].assignment)

        return qu.de_quantize_uint(
            output_values,
            self.quantization_config["quantization_bits"],
            self.quantization_config["quantization_bits"]
            - self.quantization_config["int_bits_activation"],
        )


def check_robustness_gurobi(encoding, x, y, args):  # encoding is a SMT model to solve
    """
        Return values:
            np.array: Adversarial attack if one exists
            True:     If sample is robust
            None:     If timeout or other error occured
    """
    x = x.flatten()
    x = qu.downscale_op_input(x, encoding.quantization_config)
    x = qu.fake_quant_op_input(x, encoding.quantization_config)
    x = x * (2 ** encoding.quantization_config["input_bits"])
    x = x.numpy()

    print(
        "\n============================== Begin to encode input region & output property ==============================\n")
    low, high = x - args.eps, x + args.eps
    encoding.assert_input_box(low, high)

    encoding.assert_not_argmax_gurobi(int(y))
    print(
        "\n============================== Begin to encode the QNN semantics ==============================\n")
    attack = None
    ifSat = encoding.sat(args, verbose=False)
    if ifSat:
        attack = encoding.get_input_assignment_gurobi()
        return (False, attack)
    else:
        return (True, None)
