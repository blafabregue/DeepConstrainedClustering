
"""
Keras implementation for Deep embedded Constrained Clustering :

         Zhang, H., Basu, S., \& Davidson, I. (2019). A Framework for Deep Constrained Clustering - Algorithms and Advances. ECML/PKDD 2019

Usage:
    python MLP_DEC.py fcnn Mallat 5 0.0 --archive_name Univariate --itr "0" --ae_weights "path/to/ae_weights.h5" --batch_size=16
Author:
    Baptiste Lafabregue 2019.06.20 based on work of
    Xifeng Guo. https://github.com/XifengGuo/IDEC and
    Zhang Hongjing https://github.com/blueocean92/deep_constrained_clustering
"""

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as K
import keras.layers as kl

from sklearn.cluster import KMeans
from sklearn import metrics
import math
import random
from time import time
import numpy as np

from DEC import cluster_acc, ClusteringLayer
import MLP_SDAE
import FCNN_AE
import utils


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links.

    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def init_empty_arrays():
    """
    Generate empty sets of constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    return np.array(ml_ind1, dtype=int), np.array(ml_ind2, dtype=int), \
           np.array(cl_ind1, dtype=int), np.array(cl_ind2, dtype=int)

# Define custom loss
def ml_loss():
    """
    Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    """
    def loss(y_true, y_pred):
        size = K.int_shape(y_pred)[0]
        shape = K.shape(y_pred)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // 2
        size = step
        size = K.concatenate([size, input_shape], axis=0)
        stride = K.concatenate([step, input_shape * 0], axis=0)
        start = stride * 0
        p1 = K.slice(y_pred, start, size)
        start = stride * 1
        p2 = K.slice(y_pred, start, size)
        return K.mean(-K.log(K.sum(p1 * p2, axis=1)))

    # Return a function
    return loss

# Define custom loss
def cl_loss():
    """
    Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    """
    def loss(y_true, y_pred):
        size = K.int_shape(y_pred)[0]
        shape = K.shape(y_pred)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // 2
        size = step
        size = K.concatenate([size, input_shape], axis=0)
        stride = K.concatenate([step, input_shape * 0], axis=0)
        start = stride * 0
        p1 = K.slice(y_pred, start, size)
        start = stride * 1
        p2 = K.slice(y_pred, start, size)
        return K.mean(-K.log(1.0 - K.sum(p1 * p2, axis=1)))

    # Return a function
    return loss


class CIDEC(object):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 input_dim,
                 dimensions,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(CIDEC, self).__init__()

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.input_dim = input_dim
        if classifier_name == 'fcnn':
            filters, dense_filters, kernels = dimensions
            self.conv_dims = filters
            self.dims = dense_filters
            self.n_stacks = len(self.conv_dims) - 1
            self.d_stacks = len(self.dims) - 1
            self.autoencoder = FCNN_AE.autoencoder_a2cnes(input_dim, filters, dense_filters, kernels)
        else:
            self.dims = dimensions
            self.n_stacks = len(self.dims) - 1
            self.autoencoder = MLP_SDAE.autoencoder(self.dims, input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        else:
            print('ae_weights must be given by adding path to arguments e.g. : --ae_weights weights.h5')
            exit()

        ml_penalty, cl_penalty = 0.1, 1
        cst_optimizer = 'adam'
        if self.classifier_name == 'mlp':
            hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
            decoder_layer = 'decoder_0'
        else:
            hidden = self.autoencoder.get_layer(name='encoder_d_%d' % (self.d_stacks)).output
            decoder_layer = 'lambda_o_0'

        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(loss={'clustering': 'kld', decoder_layer: 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)
        for layer in self.model.layers:
            print(layer, layer.trainable)
        self.ml_model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.ml_model.compile(loss={'clustering': ml_loss(), decoder_layer: 'mse'},
                           loss_weights=[ml_penalty, 1],
                           optimizer=cst_optimizer)
        self.cl_model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer])
        self.cl_model.compile(loss={'clustering': cl_loss()},
                           optimizer=cst_optimizer)

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def print_stats(self, x, y, x_test, y_test, loss, epoch, logwriter, prefix, stats_path=None):
        q, _ = self.model.predict(x, verbose=0)
        # evaluate the clustering performance
        y_pred = q.argmax(1)

        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        loss = np.round(loss, 5)
        logdict = dict(iter=epoch, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
        logwriter.writerow(logdict)

        # compute constraints satisfaction
        sat = 0.0
        if ml_ind1 is not None and cl_ind1 is not None and len(ml_ind1)+len(cl_ind1) > 0:
            for i in range(len(ml_ind1)):
                if y_pred[ml_ind1[i]] == y_pred[ml_ind2[i]]:
                    sat += 1.0
            for i in range(len(cl_ind1)):
                if y_pred[cl_ind1[i]] != y_pred[cl_ind2[i]]:
                    sat += 1.0
            sat /= float(len(ml_ind2) + len(cl_ind1))
        if x_test is not None and y_test is not None:
            q_test, _ = self.model.predict(x_test, verbose=0)
            # evaluate the clustering performance
            y_pred_test = q_test.argmax(1)

            acc_test = np.round(cluster_acc(y_test, y_pred_test), 5)
            nmi_test = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
            ari_test = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)
        print(prefix, ' sat: ', sat, 'ari:', ari, 'acc:', acc, 'nmi:', nmi,
              '   ###   ari_test:', ari_test, 'acc_test:', acc_test, 'nmi_test:', nmi_test)

        if stats_path is not None:
            with open(stats_path, "a+") as file:
                content = self.dataset_name+';'+prefix+';'+self.save_suffix+';'+str(sat)+';'+str(ari)+';'+str(acc)+';'+\
                      str(nmi)+';'+str(ari_test)+';'+str(acc_test)+';'+str(nmi_test)+'\n'
                file.write(content)
        return sat

    def clustering(self, x,
                   ml_ind1, ml_ind2,
                   cl_ind1, cl_ind2,
                   y=None,
                   tol=1e-3,
                   update_interval=1,
                   maxepoch=2,
                   save_dir='./results/dcc',
                   save_suffix='',
                   update_ml=3,
                   update_cl = 3,
                   x_test=None,
                   y_test=None
                ):

        self.save_suffix = save_suffix
        print('Update interval', update_interval)
        save_interval = 500  # 5 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcc_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()
        ari = 0.0
        loss = [0, 0, 0]
        max_sat = 0.0

        if y is not None:
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            print('ari kmeans: ', str(ari))
            max_sat = self.print_stats(x, y, x_test, y_test, loss, 0, logwriter, 'init')
            self.model.save_weights(save_dir + '/DCC_model_max_sat_.h5')

        # initialize the sat to have the lower limit :
        q, _ = self.model.predict(x, verbose=0)
        p = self.target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        y_pred_last = y_pred

        num_batch = int(math.ceil(1.0*x.shape[0]/self.batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/self.batch_size))
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/self.batch_size))

        epoch_iter = iter(range(int(maxepoch)))
        for epoch in epoch_iter:

            print('Epoch ', epoch)
            if epoch % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    sat = self.print_stats(x, y, x_test, y_test, loss, epoch, logwriter, 'pairwise_')
                    if sat > max_sat:
                        max_sat = sat
                        self.model.save_weights(save_dir + '/DCC_model_max_sat_.h5')

                # check stop criterion
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            for batch_idx in range(num_batch):
                if (batch_idx + 1) * self.batch_size > x.shape[0]:
                    loss = self.model.train_on_batch(x=x[batch_idx * self.batch_size::],
                                                     y=[p[batch_idx * self.batch_size::], x[batch_idx * self.batch_size::]])
                else:
                    loss = self.model.train_on_batch(x=x[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size],
                                                     y=[p[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size],
                                                        x[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]])

            if (epoch % update_cl == 0 or epoch % update_ml == 0) and (ml_num_batch+cl_num_batch > 0):
                if y is not None:
                    sat = self.print_stats(x, y, x_test, y_test, loss, epoch, logwriter, 'cluster__')
                    if sat > max_sat:
                        max_sat = sat
                        self.model.save_weights(save_dir + '/DCC_model_max_sat_.h5')

            ml_loss = 0.0
            if epoch % update_ml == 0:
                ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/self.batch_size))
                ml_num = ml_ind1.shape[0]
                for ml_batch_idx in range(ml_num_batch):
                    px1 = x[ml_ind1[ml_batch_idx*self.batch_size: min(ml_num, (ml_batch_idx+1)*self.batch_size)]]
                    px2 = x[ml_ind2[ml_batch_idx*self.batch_size: min(ml_num, (ml_batch_idx+1)*self.batch_size)]]
                    pbatch1 = p[ml_ind1[ml_batch_idx*self.batch_size: min(ml_num, (ml_batch_idx + 1)*self.batch_size)]]
                    pbatch2 = p[ml_ind2[ml_batch_idx*self.batch_size: min(ml_num, (ml_batch_idx+1)*self.batch_size)]]

                    ml_loss += self.ml_model.train_on_batch(x=np.concatenate((px1, px2), axis=0),
                                              y=[np.concatenate((pbatch1, pbatch2), axis=0),
                                                 np.concatenate((px1, px2), axis=0)])[0]

            cl_loss = 0.0
            if epoch % update_cl == 0:
                cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/self.batch_size))
                cl_num = cl_ind1.shape[0]
                for cl_batch_idx in range(cl_num_batch):
                    px1 = x[cl_ind1[cl_batch_idx*self.batch_size: min(cl_num, (cl_batch_idx+1)*self.batch_size)]]
                    px2 = x[cl_ind2[cl_batch_idx*self.batch_size: min(cl_num, (cl_batch_idx+1)*self.batch_size)]]
                    pbatch1 = p[cl_ind1[cl_batch_idx*self.batch_size: min(cl_num, (cl_batch_idx + 1)*self.batch_size)]]
                    pbatch2 = p[cl_ind2[cl_batch_idx*self.batch_size: min(cl_num, (cl_batch_idx+1)*self.batch_size)]]

                    cl_loss += self.cl_model.train_on_batch(x=np.concatenate((px1, px2), axis=0),
                                              y=[np.concatenate((pbatch1, pbatch2), axis=0)])

            if (epoch % update_cl == 0 or epoch % update_ml == 0) and (ml_num_batch+cl_num_batch > 0):
                print("Pairwise Total:", str(float(ml_loss) + float(cl_loss)), "ML loss",
                          str(ml_loss), "CL loss:", str(cl_loss))

            # save intermediate model
            if epoch % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to:', save_dir+'/DCC_model_'+str(epoch)+save_suffix+'_'+str(ari)+'.h5')
                self.model.save_weights(save_dir+'/DCC_model_'+str(epoch)+save_suffix+'_'+str(ari)+'.h5')

        # save the trained model
        print('saving model to:', save_dir + '/DCC_model_final'+save_suffix+'_'+str(ari)+'.h5')
        self.model.save_weights(save_dir + '/DCC_model_final'+save_suffix+'_'+str(ari)+'.h5')
        self.print_stats(x, y, x_test, y_test, loss, epoch, logwriter, 'final', stats_path)

        self.model.load_weights(save_dir + '/DCC_model_max_sat_.h5')
        self.print_stats(x, y, x_test, y_test, loss, epoch, logwriter, 'max_sat', stats_path)
        self.model.load_weights(save_dir + '/DCC_model_final'+save_suffix+'_'+str(ari)+'.h5')

        logfile.close()

        return y_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('type', default='mlp', choices=['mlp', 'fcnn'])
    parser.add_argument('dataset', default='mnist')

    parser.add_argument('id', default=0, type=int)
    parser.add_argument('const_perc', default=0, type=float)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--maxepoch', default=200, type=int)
    parser.add_argument('--archive_name', default='Univariate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None, help='This argument must be given')
    parser.add_argument('--dimensions', default=None, type=str)
    parser.add_argument('--itr', default='0')
    parser.add_argument('--stats_path', default=None, type=str)
    args = parser.parse_args()
    print(args)

    # load dataset
    optimizer = SGD(lr=0.1, momentum=0.9)

    archive_name = args.archive_name
    dataset_name = args.dataset
    classifier_name = args.type
    itr = args.itr
    root_dir = '.'
    z = 10
    stats_path = args.stats_path

    train_dict = utils.read_multivariate_dataset(root_dir, archive_name, dataset_name, True)
    x = train_dict[dataset_name][0]
    y = train_dict[dataset_name][1]

    train_dict_test = utils.read_multivariate_dataset(root_dir, archive_name, dataset_name, False)
    x_test = train_dict_test[dataset_name][0]
    y_test = train_dict_test[dataset_name][1]

    optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-6)
    clust_nb = train_dict['k']

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/' + \
                       dataset_name + '/' + itr + "/"
    output_directory = utils.create_directory(output_directory)
    const_perc = args.const_perc
    constraints_size = int(len(x)*const_perc)
    suffix = '_'+str(args.maxepoch)+'_'+str(z)+'_'+str(args.id)+'_const'+str(constraints_size)

    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = init_empty_arrays()
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(y, constraints_size)
    # if constraints_size > 0:
    #     ml_ind1, ml_ind2, cl_ind1, cl_ind2 = utils.read_a2cnes_constraints(root_dir, archive_name, dataset_name, const_perc, args.id)
    # else:
    #     ml_ind1, ml_ind2, cl_ind1, cl_ind2 = init_empty_arrays()
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, x.shape[0])
    ml_penalty, cl_penalty = 0.1, 1

    # prepare the IDEC model
    if args.dimensions is None:
        if classifier_name == 'mlp':
            dimensions = [x.shape[1]*x.shape[2], 500, 500, 2000, z]
        elif classifier_name == 'fcnn':
            dimensions = ([128, 256, 128], [z], [9, 5, 3])
    else:
        if classifier_name == 'mlp':
            dimensions = [x.shape[1]*x.shape[2]] + eval(args.dimensions)
        elif classifier_name == 'fcnn':
            dimensions = eval(args.dimensions)

    idec = CIDEC(dataset_name, classifier_name, x.shape[1:], dimensions=dimensions,
                n_clusters=clust_nb, batch_size=args.batch_size)
    idec.initialize_model(ae_weights=args.ae_weights, gamma=args.gamma, optimizer=optimizer)
    # plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = idec.clustering(x, ml_ind1, ml_ind2, cl_ind1, cl_ind2, y=y, tol=args.tol,
                             maxepoch=args.maxepoch, update_interval=1,
                             save_dir=output_directory, save_suffix=suffix, update_ml=1, update_cl=1,
                             x_test=x_test, y_test=y_test)
    ari = metrics.adjusted_rand_score(y, y_pred)
    print('ari:', ari)
    print('clustering time: ', (time() - t0))
