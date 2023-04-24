import copy
import pandas
import numpy as np
import utilities


def emt_preprocessing():

    emt_dataframe = pandas.read_csv("data/measures_v2.csv")
    emt_data = emt_dataframe.to_numpy()

    # We eliminate the profile ID feature.
    emt_data = emt_data[:, :-1]

    # For our dataset class we need to reorder the columns so that the potential targets are all at the end.
    # Column 11 will be torque, Column 10 will be stator_yoke (c9) and Column 9 will be stator_tooth (c4).
    temp_col = copy.deepcopy(emt_data[:, 10])
    emt_data[:, 10] = emt_data[:, 9]
    emt_data[:, 9] = emt_data[:, 4]
    emt_data[:, 4] = temp_col

    ubs = np.max(emt_data, 0)
    lbs = np.min(emt_data, 0)

    # We normalize the data for the network using the sigmoid activation function between 0 and 1
    sig_norm = utilities.Normalize(lbs, ubs - lbs)
    # and between -1 and 1 for the tanh activation function.
    tanh_norm = utilities.Normalize((ubs + lbs) / 2, (ubs - lbs) / 2)

    sig_data = sig_norm(emt_data)
    tanh_data = tanh_norm(emt_data)
    np.savetxt("data/emt_data_sig.csv", sig_data, delimiter=",", fmt='%.15f')
    np.savetxt("data/emt_data_tanh.csv", tanh_data, delimiter=",", fmt='%.15f')


emt_preprocessing()
