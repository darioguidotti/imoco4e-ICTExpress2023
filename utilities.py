import pynever.networks as pyn_networks
import pynever.strategies.verification as pyn_verification
import pynever.strategies.conversion as pyn_conversion
import torch
import pynever.nodes as pyn_nodes
from pynever.tensor import Tensor
import numpy as np
from typing import Callable


class Normalize(Callable):

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def __call__(self, data: np.ndarray):
        return (data - self.mean) / self.std


def generate_lrobustness_property(network: pyn_networks.SequentialNetwork, input_sample: Tensor, epsilon: float,
                                  delta: float, lbs: Tensor = None, ubs: Tensor = None) -> pyn_verification.NeVerProperty:

    pyt_net = pyn_conversion.PyTorchConverter().from_neural_network(network).pytorch_network
    pyt_net.double()
    in_pyt_sample = torch.from_numpy(input_sample).double()
    output_sample = pyt_net(in_pyt_sample).detach().numpy()

    in_pred_mat = []
    in_pred_bias = []
    input_size = len(input_sample)
    for i in range(input_size):

        lb_constraint = np.zeros(input_size)
        ub_constraint = np.zeros(input_size)
        lb_constraint[i] = -1
        ub_constraint[i] = 1
        in_pred_mat.append(lb_constraint)
        in_pred_mat.append(ub_constraint)
        if lbs is not None and input_sample[i] - epsilon < lbs[i]:
            in_pred_bias.append([-lbs[i]])
        else:
            in_pred_bias.append([-(input_sample[i] - epsilon)])

        if ubs is not None and input_sample[i] + epsilon > ubs[i]:
            in_pred_bias.append([ubs[i]])
        else:
            in_pred_bias.append([input_sample[i] + epsilon])

    in_pred_bias = np.array(in_pred_bias)
    in_pred_mat = np.array(in_pred_mat)

    output_size = len(output_sample)
    out_pred_mat = []
    out_pred_bias = []
    for i in range(output_size):
        lb_constraint = np.zeros((1, output_size))
        ub_constraint = np.zeros((1, output_size))
        lb_constraint[0][i] = 1
        ub_constraint[0][i] = -1
        out_pred_mat.append(lb_constraint)
        out_pred_mat.append(ub_constraint)
        out_pred_bias.append(np.array([[(output_sample[i] - delta)]]))
        out_pred_bias.append(np.array([[-(output_sample[i] + delta)]]))

    return pyn_verification.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)


def generate_advrobustness_property(fex_network: pyn_networks.SequentialNetwork,
                                    cls_network: pyn_networks.SequentialNetwork,
                                    input_sample: Tensor, epsilon: float) -> pyn_verification.NeVerProperty:

    pyt_fex_net = pyn_conversion.PyTorchConverter().from_neural_network(fex_network).pytorch_network
    pyt_cls_net = pyn_conversion.PyTorchConverter().from_neural_network(cls_network).pytorch_network
    pyt_fex_net.float()
    pyt_cls_net.float()
    cls_input_sample = pyt_fex_net(torch.from_numpy(input_sample)).detach().numpy()
    cls_output = pyt_cls_net(torch.from_numpy(cls_input_sample)).detach().squeeze().numpy()
    cls_input_sample = cls_input_sample.squeeze()
    correct_target = np.argmax(cls_output)

    in_pred_mat = []
    in_pred_bias = []
    input_size = len(cls_input_sample)
    for i in range(input_size):

        lb_constraint = np.zeros(input_size)
        ub_constraint = np.zeros(input_size)
        lb_constraint[i] = -1
        ub_constraint[i] = 1

        in_pred_mat.append(lb_constraint)
        in_pred_mat.append(ub_constraint)
        in_pred_bias.append([-(cls_input_sample[i] - epsilon)])
        in_pred_bias.append([cls_input_sample[i] + epsilon])

    in_pred_bias = np.array(in_pred_bias)
    in_pred_mat = np.array(in_pred_mat)

    out_pred_mat = []
    out_pred_bias = []
    output_size = len(cls_output)
    for i in range(output_size):

        if i != correct_target:
            max_class_constraint = np.zeros((1, output_size))
            max_class_constraint[0][correct_target] = 1
            max_class_constraint[0][i] = -1
            out_pred_mat.append(max_class_constraint)
            out_pred_bias.append(np.array([[0]]))

    return pyn_verification.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)


def to_smtlib(network: pyn_networks.SequentialNetwork, prop: pyn_verification.NeVerProperty, filepath: str,
              input_prefix: str = "X", output_prefix: str = "Y", hidden_prefix: str = "H", smt_solver: str = "CVC5"):

    with open(filepath, "w+") as f:

        if smt_solver == "CVC5":
            f.write("(set-logic QF_NRAT)\n\n")
        else:
            f.write("(set-logic QF_NRA)\n\n")

        has_sigmoid = False
        has_tanh = False
        has_relu = False

        current_node = network.get_first_node()
        while current_node is not None:
            if isinstance(current_node, pyn_nodes.SigmoidNode):
                has_sigmoid = True
            elif isinstance(current_node, pyn_nodes.TanhNode):
                has_tanh = True
            elif isinstance(current_node, pyn_nodes.ReLUNode):
                has_relu = True
            current_node = network.get_next_node(current_node)

        if has_sigmoid:
            f.write(";; --- SIGMOID DEFINITION ---\n")
            f.write("(define-fun sigmoid ((x Real)) Real (/ 1 (+ (exp (- x)) 1)))\n\n")

        if has_tanh:
            f.write(";; --- TANH DEFINITION ---\n")
            f.write("(define-fun tanh ((x Real)) Real (/ (- (exp x) (exp (- x))) (+ (exp x) (exp (- x)))))\n\n")

        if has_relu:
            f.write(";; --- MAX DEFINITION ---\n")
            f.write("(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))\n\n")

    prop.to_smt_file(filepath=filepath)

    with open(filepath, "a+") as f:

        f.write("\n\n")
        layer_index = 1
        current_node = network.get_first_node()
        while current_node is not None:

            if layer_index == 1:
                current_input_ids = [f"{input_prefix}_{i}" for i in range(current_node.in_dim[0])]
            else:
                current_input_ids = [f"{hidden_prefix}_{layer_index - 1}_{i}" for i in range(current_node.in_dim[0])]

            if layer_index == network.num_nodes:
                current_output_ids = [f"{output_prefix}_{i}" for i in range(current_node.out_dim[0])]
            else:
                current_output_ids = [f"{hidden_prefix}_{layer_index}_{i}" for i in range(current_node.out_dim[0])]

            f.write(f";; --- LAYER {current_node.__str__()} ---\n\n")

            num_hidden_v = current_node.out_dim[0]
            if layer_index != network.num_nodes:
                f.write(f";; --- HIDDEN VARIABLES ---\n")
                for i in range(num_hidden_v):
                    f.write(f"(declare-fun {current_output_ids[i]} () Real)\n")
                f.write("\n")

            f.write(f";; --- CONSTRAINTS ---\n")
            if isinstance(current_node, pyn_nodes.FullyConnectedNode):

                for i in range(num_hidden_v):
                    constraint = f"(assert (= {current_output_ids[i]} (+"
                    for j in range(current_node.weight.shape[1]):
                        if current_node.weight[i][j] < 0:
                            constraint += f" (* {current_input_ids[j]} (- {abs(current_node.weight[i][j])}))"
                        else:
                            constraint += f" (* {current_input_ids[j]} {current_node.weight[i][j]})"

                    if current_node.has_bias:
                        if current_node.bias[i] < 0:
                            constraint += f" (- {abs(current_node.bias[i])})"
                        else:
                            constraint += f" {current_node.bias[i]}"

                    constraint += ")))\n"
                    f.write(constraint)
                f.write("\n")

            elif isinstance(current_node, pyn_nodes.SigmoidNode):

                for i in range(num_hidden_v):
                    constraint = f"(assert (= {current_output_ids[i]} (sigmoid {current_input_ids[i]})))\n"
                    f.write(constraint)
                f.write("\n")

            elif isinstance(current_node, pyn_nodes.TanhNode):

                for i in range(num_hidden_v):
                    constraint = f"(assert (= {current_output_ids[i]} (tanh {current_input_ids[i]})))\n"
                    f.write(constraint)
                f.write("\n")

            elif isinstance(current_node, pyn_nodes.ReLUNode):

                for i in range(num_hidden_v):
                    constraint = f"(assert (= {current_output_ids[i]} (max {current_input_ids[i]} 0)))\n"
                    f.write(constraint)
                f.write("\n")

            current_node = network.get_next_node(current_node)
            layer_index += 1

        f.write("(check-sat)\n")
        f.write("(exit)\n")


def to_smtlib_no_aux_var(network: pyn_networks.SequentialNetwork, prop: pyn_verification.NeVerProperty, filepath: str,
                         input_prefix: str = "X", output_prefix: str = "Y", hidden_prefix: str = "H",
                         smt_solver: str = "Mathsat"):

    with open(filepath, "w+") as f:

        if smt_solver == "CVC5":
            f.write("(set-logic QF_NRAT)\n\n")
        else:
            f.write("(set-logic QF_NRA)\n\n")

        has_sigmoid = False
        has_tanh = False
        has_relu = False

        current_node = network.get_first_node()
        while current_node is not None:
            if isinstance(current_node, pyn_nodes.SigmoidNode):
                has_sigmoid = True
            elif isinstance(current_node, pyn_nodes.TanhNode):
                has_tanh = True
            elif isinstance(current_node, pyn_nodes.ReLUNode):
                has_relu = True
            current_node = network.get_next_node(current_node)

        if has_sigmoid:
            f.write(";; --- SIGMOID DEFINITION ---\n")
            f.write("(define-fun sigmoid ((x Real)) Real (/ 1 (+ (exp (- x)) 1)))\n\n")

        if has_tanh:
            f.write(";; --- TANH DEFINITION ---\n")
            f.write("(define-fun tanh ((x Real)) Real (/ (- (exp x) (exp (- x))) (+ (exp x) (exp (- x)))))\n\n")

        if has_relu:
            f.write(";; --- MAX DEFINITION ---\n")
            f.write("(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))\n\n")

    prop.to_smt_file(filepath=filepath)

    with open(filepath, "a+") as f:

        f.write("\n\n")
        layer_index = 1
        current_node = network.get_first_node()
        current_input_ids = [f"{input_prefix}_{i}" for i in range(current_node.in_dim[0])]
        output_ids = [f"{output_prefix}_{i}" for i in range(network.get_last_node().out_dim[0])]

        f.write(f";; --- NETWORK ENCODING ---\n")
        while current_node is not None:

            temp_ids = []
            if isinstance(current_node, pyn_nodes.FullyConnectedNode):

                for i in range(current_node.weight.shape[0]):

                    constraint = f"(+"
                    for j in range(current_node.weight.shape[1]):
                        if current_node.weight[i][j] < 0:
                            constraint += f" (* {current_input_ids[j]} (- {abs(current_node.weight[i][j]):.18f}))"
                        else:
                            constraint += f" (* {current_input_ids[j]} {current_node.weight[i][j]:.18f})"

                    if current_node.has_bias:
                        if current_node.bias[i] < 0:
                            constraint += f" (- {abs(current_node.bias[i]):.18f})"
                        else:
                            constraint += f" {current_node.bias[i]:.18f}"

                    constraint += ")"
                    temp_ids.append(constraint)

            elif isinstance(current_node, pyn_nodes.SigmoidNode):

                for i in range(current_node.out_dim[0]):
                    constraint = f"(sigmoid {current_input_ids[i]})"
                    temp_ids.append(constraint)

            elif isinstance(current_node, pyn_nodes.TanhNode):

                for i in range(current_node.out_dim[0]):
                    constraint = f"(tanh {current_input_ids[i]})"
                    temp_ids.append(constraint)

            elif isinstance(current_node, pyn_nodes.ReLUNode):

                for i in range(current_node.out_dim[0]):
                    constraint = f"(max {current_input_ids[i]} 0)"
                    temp_ids.append(constraint)

            current_input_ids = temp_ids
            current_node = network.get_next_node(current_node)
            layer_index += 1

        for i in range(len(output_ids)):
            constraint = f"(assert (= {output_ids[i]} {current_input_ids[i]}))\n"
            f.write(constraint)
        f.write("\n")

        f.write("(check-sat)\n")
        f.write("(exit)\n")
