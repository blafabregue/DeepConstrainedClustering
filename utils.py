import numpy as np
import matplotlib
import os
import csv

import scipy.interpolate as interpolate

# matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y


def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return directory_path
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

def create_path(root_dir,classifier_name, archive_name):
    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+'/'
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

def write_clustering(y, file_path):
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for l in y:
            writer.writerow([l])


def read_a2cnes_constraints(root_dir, archive_name, dataset_name, size, id):
    dict = {}

    data_file = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/test/' +\
                dataset_name + '_' + str(size) + '_' + str(id) + '.constraints'
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in reader:
            tmp1 = int(row[0])-1
            tmp2 = int(row[1])-1
            if int(row[2]) == 1:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
            else:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)

    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def read_multivariate_dataset(root_dir, archive_name, dataset_name, is_train):
    dict = {}

    if is_train:
        type = '/train/'
    else:
        type = '/test/'

    data_file = root_dir+'/archives/'+archive_name+'/'+dataset_name+type+dataset_name+'.data'
    label_file = root_dir+'/archives/'+archive_name+'/'+dataset_name+type+dataset_name+'.labels'
    feature_file = root_dir+'/archives/'+archive_name+'/'+dataset_name+type+dataset_name+'.f'
    k_file = root_dir+'/archives/'+archive_name+'/'+dataset_name+type+dataset_name+'.k'

    x = []
    y = []
    sep = '\t'

    with open(feature_file, 'r') as file:
        feature_count = int(file.read())

    with open(k_file, 'r') as file:
        k = int(file.read())

    with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=sep, quotechar='"')
        for row in reader:
            elem = []
            length = int(len(row) / feature_count)
            for i in range(length):
                t = []
                for f in range(feature_count):
                    t.append(float(row[i*feature_count+f]))
                elem.append(t)
            x.append(elem)

    with open(label_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=sep, quotechar='"')
        for row in reader:
            y.append(int(row[0]))

    dict[dataset_name] = (np.array(x), np.array(y))
    dict['k'] = k

    return dict




def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n=x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n=x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx= np.array(range(curr_length))
        idx_new = np.linspace(0,idx.max(),max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            # changed from this version
            # new_ts = spline(idx,ts,idx_new)
            tck = interpolate.splrep(x, ts, s=0)
            new_ts = interpolate.splev(idx_new, tck, der=0)
            ucr_x[i, : j] = new_ts

    return ucr_x


