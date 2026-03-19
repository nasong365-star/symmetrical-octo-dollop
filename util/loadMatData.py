import random
import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ratio, seed=42):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled

def load_data_2(dataset, path="./datasets/"):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    feature_list = []

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature_list.append(feature)
    return feature_list, labels

def load_data_single(dataset, path="./datasets/"):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    adj = data['adj']
    features = normalize(features)
    if ss.isspmatrix_csr(features):
        features = features.todense()
        print("sparse")
    return features, labels, adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_to_Lap(features, knns=10):
    lap = []
    sim = []
    adj = []
    lap2 = []
    for i in range(len(features)):
        tt = kneighbors_graph(features[i], knns, mode="distance")
        tt = sp.coo_matrix(tt)
        tt = tt + tt.T.multiply(tt.T > tt) - tt.multiply(tt.T > tt)
        temp = kneighbors_graph(features[i].T, knns, mode="distance")
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        adj.append(sparse_mx_to_torch_sparse_tensor(tt))
        sim.append(sparse_mx_to_torch_sparse_tensor(temp))
        lap.append(sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp)))
        lap2.append(sparse_mx_to_torch_sparse_tensor(construct_laplacian(tt)))
    return sim, lap, adj, lap2


def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    # adj_ = sp.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    # print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = sp.eye(adj.shape[0]) - adj_wave
    lp = adj_wave
    return lp


