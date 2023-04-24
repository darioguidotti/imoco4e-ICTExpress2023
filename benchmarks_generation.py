import os.path

import onnx
import pynever.strategies.training as pyn_training
import pynever.strategies.conversion as pyn_conversion
import pynever.datasets as pyn_data
import pynever.networks as pyn_net
import pynever.nodes as pyn_nodes
import numpy as np
import torch.utils.data as tdt
import torch.optim as topt
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import utilities

from datetime import datetime


# ===== SET EXPERIMENT ID AND FOLDERS CREATION =====
#
#
#
benchmark_datetime = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
experiment_folder = f"benchmarks_{benchmark_datetime}/"
onnx_folder = experiment_folder + "onnx_models/"
smtlib_folder = experiment_folder + "smtlib_benchmarks/"
logs_folder = experiment_folder + "logs/"
checkpoint_folder = experiment_folder + "training_checkpoints/"

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(onnx_folder):
    os.mkdir(onnx_folder)

if not os.path.exists(smtlib_folder):
    os.mkdir(smtlib_folder)

if not os.path.exists(logs_folder):
    os.mkdir(logs_folder)

if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)


# ===== LOGGERS INSTANTIATION =====
#
#
#
logger_stream = logging.getLogger("pynever.strategies.training")
logger_file = logging.getLogger("benchmark_generation_file")

file_handler = logging.FileHandler(f"{logs_folder}benchmark_gen_logs.txt")
stream_handler = logging.StreamHandler()

file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)

logger_file.addHandler(file_handler)
logger_stream.addHandler(stream_handler)

logger_file.setLevel(logging.INFO)
logger_stream.setLevel(logging.INFO)

#
#
#
# ===== PARAMETERS SELECTION =====
#
#
#
device = "mps"
emt_benchmark_parameters = {

    # DATASET PARAMETERS
    "dataset_id": "emt",
    "dataset_folder": "data/",
    "first_target_index": 9,

    # NETWORK PARAMETERS
    "net_archs": [[16], [32], [16, 8], [32, 16], [64]],
    "activation_functions": [pyn_nodes.ReLUNode, pyn_nodes.SigmoidNode, pyn_nodes.TanhNode],
    # "activation_functions": [pyn_nodes.ReLUNode, pyn_nodes.SigmoidNode, pyn_nodes.TanhNode],
    "input_dimension": (9,),
    "output_size": 3,
    "activation_on_output": False,

    # TRAINING PARAMETERS
    "validation_percentage": 0.3,
    "loss_fn": nn.MSELoss(),
    "n_epochs": 5,
    "train_batch_size": 1024,
    "validation_batch_size": 512,
    "opt_con": topt.Adam,
    "opt_params": {"lr": 0.001, "weight_decay": 0},

    # TESTING PARAMETERS
    "test_percentage": 0.2,
    "output_labels": ["Motor Torque", "Stator Yoke Temperature", "Stator Tooth Temperature"],
    "num_sample_plot": 100,
    "test_batch_size": 1024,
    "save_results": False,
    "metric": nn.MSELoss(),
    "metric_params": {},

    # PROPERTY PARAMETERS
    "epsilons": [0.001, 0.01, 0.1],
    "deltas": [0.1, 1],

}

benchmarks_parameters = [emt_benchmark_parameters]
smt_aux_vars = False

logger_file.info(f"benchmark_id,"
                 f"datetime,"
                 f"dataset_id,"
                 f"net_arch,"
                 f"activation_function,"
                 f"activation_on_output,"
                 f"validation_percentage,"
                 f"loss_fn,"
                 f"n_epochs,"
                 f"train_batch_size,"
                 f"validation_batch_size,"
                 f"optimizer,"
                 f"lr,"
                 f"weight_decay,"
                 f"test_percentage,"
                 f"test_batch_size,"
                 f"metric,"
                 f"test_metric_results,"
                 f"epsilon,"
                 f"delta")

#
#
#
# ===== BENCHMARK GENERATION =====
#
#
#
benchmark_num = 0
for benchmark_params in benchmarks_parameters:

    # DATASET PARAMETERS
    dataset_id = benchmark_params["dataset_id"]
    dataset_folder = benchmark_params["dataset_folder"]
    first_target_index = benchmark_params["first_target_index"]

    # NETWORK PARAMETERS
    net_archs = benchmark_params["net_archs"]
    activation_functions = benchmark_params["activation_functions"]
    input_dimension = benchmark_params["input_dimension"]
    output_size = benchmark_params["output_size"]
    activation_on_output = benchmark_params["activation_on_output"]

    # TRAINING PARAMETERS
    validation_percentage = benchmark_params["validation_percentage"]
    loss_fn = benchmark_params["loss_fn"]
    n_epochs = benchmark_params["n_epochs"]
    train_batch_size = benchmark_params["train_batch_size"]
    validation_batch_size = benchmark_params["validation_batch_size"]
    checkpoint_root = checkpoint_folder
    opt_con = benchmark_params["opt_con"]
    opt_params = benchmark_params["opt_params"]

    # TESTING PARAMETERS
    test_percentage = benchmark_params["test_percentage"]
    output_labels = benchmark_params["output_labels"]
    num_sample_plot = benchmark_params["num_sample_plot"]
    test_batch_size = benchmark_params["test_batch_size"]
    save_results = benchmark_params["save_results"]
    metric = benchmark_params["metric"]
    metric_params = benchmark_params["metric_params"]

    # PROPERTY PARAMETERS
    epsilons = benchmark_params["epsilons"]
    deltas = benchmark_params["deltas"]

    logger_stream.info(f"BENCHMARKS {dataset_id}")

    for act_fun in activation_functions:

        if act_fun == pyn_nodes.SigmoidNode or act_fun == pyn_nodes.ReLUNode:
            dataset_path = dataset_folder + dataset_id + "_data_sig.csv"
            lbs = np.zeros(input_dimension)
            ubs = np.ones(input_dimension)
        else:
            dataset_path = dataset_folder + dataset_id + "_data_tanh.csv"
            lbs = -np.ones(input_dimension)
            ubs = np.ones(input_dimension)

        dataset = pyn_data.GenericFileDataset(dataset_path, first_target_index)
        input_sample = dataset.__getitem__(0)[0]
        test_len = int(np.floor(dataset.__len__() * test_percentage))
        train_len = dataset.__len__() - test_len
        training_dataset, test_dataset = tdt.random_split(dataset, (train_len, test_len))
        logger_stream.info(f"Training Dataset Size: {train_len}")
        logger_stream.info(f"Test Dataset Size: {test_len}")
        logger_stream.info("")

        for net_arch in net_archs:

            net_id = f"{dataset_id}_{act_fun.__name__}_{net_arch}"
            network = pyn_net.SequentialNetwork(identifier=net_id, input_id="X")

            node_index = 0
            in_dim = input_dimension
            for n_neurons in net_arch:

                fc_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=n_neurons)
                network.add_node(fc_node)
                node_index += 1

                act_node = act_fun(identifier=f"ACT_{node_index}", in_dim=fc_node.out_dim)
                network.add_node(act_node)
                in_dim = act_node.out_dim
                node_index += 1

            fc_out_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=output_size)
            network.add_node(fc_out_node)
            if activation_on_output:

                act_out_node = act_fun(identifier=f"ACT_{node_index}", in_dim=fc_out_node.out_dim)
                network.add_node(act_out_node)

            train_strategy = pyn_training.PytorchTraining(optimizer_con=opt_con, opt_params=opt_params,
                                                          loss_function=loss_fn, n_epochs=n_epochs,
                                                          validation_percentage=validation_percentage,
                                                          train_batch_size=train_batch_size,
                                                          validation_batch_size=validation_batch_size,
                                                          checkpoints_root=checkpoint_root)

            network = train_strategy.train(network=network, dataset=training_dataset)

            test_strategy = pyn_training.PytorchTesting(metric=metric, metric_params=metric_params,
                                                        test_batch_size=test_batch_size,
                                                        save_results=save_results)
            loss = test_strategy.test(network, test_dataset)
            logger_stream.info(f"Test Loss: {loss}")

            if save_results:
                outputs = np.array(test_strategy.outputs).squeeze()
                targets = np.array(test_strategy.targets).squeeze()
                losses = np.array(test_strategy.losses)
                indexes = np.arange(0, len(outputs), int(len(outputs) / num_sample_plot))

                for i in range(3):
                    plt.plot(outputs[indexes, i], label=output_labels[i] + " Outputs")
                    plt.plot(targets[indexes, i], label=output_labels[i] + " Targets")
                    plt.legend()

            # SAVE ONNX MODEL
            onnx_path = onnx_folder + network.identifier + ".onnx"
            onnx_net = pyn_conversion.ONNXConverter().from_neural_network(network).onnx_network
            onnx.save(onnx_net, onnx_path)

            # GENERATE SMTLIB PROPERTIES
            for epsilon in epsilons:
                for delta in deltas:

                    benchmark_id = f"B_{benchmark_num:03d}"
                    sanified_arch = str(net_arch).replace(", ", "-")
                    logger_file.info(
                        f"{benchmark_id},{benchmark_datetime},{dataset_id},{sanified_arch},{act_fun.__name__},{activation_on_output},"
                        f"{validation_percentage},"
                        f"{loss_fn.__class__.__name__},{n_epochs},{train_batch_size},{validation_batch_size},"
                        f"{opt_con.__name__},{opt_params['lr']},{opt_params['weight_decay']},"
                        f"{test_percentage},{test_batch_size},{metric},{loss},{epsilon},{delta}")

                    smtlib_path_cvc = smtlib_folder + f"{benchmark_id}_cvc.smt2"
                    smtlib_path_mathsat = smtlib_folder + f"{benchmark_id}_mathsat.smt2"

                    net_property = utilities.generate_lrobustness_property(network, input_sample, epsilon,
                                                                           delta, lbs, ubs)

                    if smt_aux_vars:
                        utilities.to_smtlib(network, net_property, smtlib_path_cvc, smt_solver="CVC5")
                        utilities.to_smtlib(network, net_property, smtlib_path_mathsat, smt_solver="Mathsat")
                    else:
                        utilities.to_smtlib_no_aux_var(network, net_property, smtlib_path_cvc, smt_solver="CVC5")
                        utilities.to_smtlib_no_aux_var(network, net_property, smtlib_path_mathsat, smt_solver="Mathsat")

                    benchmark_num += 1
