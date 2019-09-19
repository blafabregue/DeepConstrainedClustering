from sktime.utils.load_data import load_from_tsfile_to_dataframe
import os
import numpy as np
import sys
from dtw import dtw
import random
import csv

euclidean_norm = lambda x, y: np.linalg.norm(x-y)

def write_dataset(path, df, df_min, df_diff, max_length, distances_path):

    values = []
    with open(path, 'w+', newline='') as file:
        for index, row in df.iterrows():
            val = []
            res = ''
            for i in range(len(row[0].values)):
                val_t = []
                for id in row.index:
                    val_t.append((row[id].values[i] - df_min[id]) / df_diff[id])
                    res += str(val_t[-1]) + '\t'
                val.append(np.array(val_t))
            for i in range(max_length - len(row[0].values)):
                val_t = []
                for id in row.index:
                    val_t.append(0.0)
                    res += str(0.0) + '\t'
                val.append(np.array(val_t))

            res = res[:-1] + '\n'
            file.write(res)
            values.append(np.array(val))


def generate_constraint_list_by_data_points(labels, percent):
    ml_symbol = 1
    cl_symbol = -1

    n = len(labels)

    nC = int(round(n * percent))  # fraction of number of points

    if nC <= 0:
        nC = 1

    constraints = []

    for i in range(nC):
        # Get indexes of two distinct samples
        idx = random.sample(range(n), 2)

        if labels[idx[0]] == labels[idx[1]]:
            # ML constraint
            idx.append(ml_symbol)
        else:
            # CL constraint
            idx.append(cl_symbol)

        if i > 1:
            # Check that they are not already linked by a constraint

            while idx in constraints:
                idx = random.sample(range(n), 2)

                if labels[idx[0]] == labels[idx[1]]:
                    # ML constraint
                    idx.append(ml_symbol)
                else:
                    # CL constraint
                    idx.append(cl_symbol)

        # Add 1 to index to match matlab format
        idx[0] += 1
        idx[1] += 1
        constraints.append(idx)

    return constraints

# change here the name of the extr
archive = "Univariate2018_ts"

for root, dirs, files in os.walk("./"+archive+"/"):
    for x in dirs:
        dataset = "./"+archive+"/" + x + "/" + x
        print(x)
        if not os.path.isdir("./"+archive+"_a2cnes/" + x):
            print(' --- > create')
            train_x, train_y = load_from_tsfile_to_dataframe(dataset + "_TRAIN.ts")
            test_x, test_y = load_from_tsfile_to_dataframe(dataset + "_TEST.ts")

            # compute min, max, for, normalization
            df_max = dict([(i, sys.float_info.min) for i in train_x.iloc[0].index])
            df_min = dict([(i, sys.float_info.max) for i in train_x.iloc[0].index])
            max_length = 0
            for index, row in train_x.iterrows():
                for id in row.index:
                    max_ = row[id].max()
                    if max_ > df_max[id]:
                        df_max[id] = max_
                    min_ = row[id].min()
                    if min_ < df_min[id]:
                        df_min[id] = min_
                if max_length < row[0].size:
                    max_length = row[0].size
            df_diff = df_max.copy()
            for k in df_diff.keys():
                df_diff[k] -= df_min[k]

            os.mkdir("./"+archive+"_a2cnes/" + x)

            train_dir = "./"+archive+"_a2cnes/" + x + "/train/"
            os.mkdir(train_dir)

            write_dataset(train_dir + x + ".data", train_x, df_min, df_diff, max_length, train_dir + x + ".distances")

            unique = np.unique(train_y)
            with open(train_dir + x + ".u", 'w+', newline='') as file:
                file.write(str(unique))

            with open(train_dir + x + ".labels", 'w+', newline='') as file:
                for i in range(len(unique)):
                    train_y[train_y == unique[i]] = i
                for i in train_y:
                    file.write(str(int(i))+'\n')

            test_dir = "./"+archive+"_a2cnes/" + x + "/test/"
            os.mkdir(test_dir)

            write_dataset(test_dir + x + ".data", test_x, df_min, df_diff, max_length, test_dir + x + ".distances")

            with open(test_dir + x + ".labels", 'w+', newline='') as file:
                for i in range(len(unique)):
                    test_y[test_y == unique[i]] = i
                for i in test_y:
                    file.write(str(int(i))+'\n')

            with open(test_dir + x + ".k", 'w+', newline='') as file:
                file.write(str(len(np.unique(test_y))))
            with open(test_dir + x + ".f", 'w+', newline='') as file:
                file.write(str(train_x.shape[1]))

            with open(train_dir + x + ".k", 'w+', newline='') as file:
                file.write(str(len(np.unique(train_y))))
            with open(train_dir + x + ".f", 'w+', newline='') as file:
                file.write(str(train_x.shape[1]))

            constraint_fractions = [0.01, 0.05, 0.15, 0.5]  # percentage of points in dataset

            for c in range(len(constraint_fractions)):
                for l in range(10):

                    # if c == 1:
                    #     print("set : " + str(c) + '\n')

                    # Construct constraint list
                    constraints = generate_constraint_list_by_data_points(train_y, constraint_fractions[c])
                    # print('Total number of constraints: ' + str(len(constraints)) + '\n')

                    csv_filename = train_dir + x + '_' + str(constraint_fractions[c]) + '_' + str(
                        l) + '.constraints'
                    with open(csv_filename, 'w+') as f:
                        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                        writer.writerows(constraints)
                    # for line in constraints:

            for c in range(len(constraint_fractions)):
                for l in range(10):

                    # if c == 1:
                    #     print("set : " + str(c) + '\n')

                    # Construct constraint list
                    constraints = generate_constraint_list_by_data_points(test_y, constraint_fractions[c])
                    # print('Total number of constraints: ' + str(len(constraints)) + '\n')

                    csv_filename = test_dir + x + '_' + str(constraint_fractions[c]) + '_' + str(
                        l) + '.constraints'
                    with open(csv_filename, 'w+') as f:
                        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                        writer.writerows(constraints)
                    # for line in constraints:
