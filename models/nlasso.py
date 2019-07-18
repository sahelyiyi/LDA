import requests
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import re
import argparse
import math

from sklearn.neighbors import NearestNeighbors


def create_links(X_train, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_train)
    _, indices = nbrs.kneighbors(X_train)
    return indices


def block_clipping(weights_in, feature_dim, nr_edges, _lambda):
    mtx = np.reshape(weights_in, (feature_dim, nr_edges), order="F")

    x_norm = np.sqrt(sum(np.power(mtx, 2)))

    idx_exceed = np.where(x_norm > _lambda)[0]
    factor = np.ones(nr_edges)

    for iter_idx in range(len(idx_exceed)):
        idx = idx_exceed[iter_idx]
        factor[idx] = np.divide(_lambda, x_norm[idx])

    weights_out = np.dot(mtx, np.diag(factor))
    weights_out = np.reshape(weights_out, (feature_dim*nr_edges), order="F")

    return weights_out


def update_x_linreg(weights_in, sampling_set, feature_dim, nr_nodes, mtx_A, vec_B):
    tmp = np.dot(mtx_A, weights_in) + vec_B
    tmp = np.reshape(tmp, (feature_dim, nr_nodes), order="F")

    weights_out = np.reshape(weights_in, (feature_dim, nr_nodes), order="F")
    weights_out[:, sampling_set] = tmp[:, sampling_set]
    weights_out = np.reshape(weights_out, (feature_dim * nr_nodes), order="F")

    return weights_out


def lad(X, y, yerr=None, l1_regularizer=0., maxiter=50, rtol=1e-4,
        eps=1e-4, session=None):
    if yerr is not None:
        whitening_factor = yerr/math.sqrt(2.)
    else:
        whitening_factor = 1.

    # converts inputs to tensors
    X_tensor = tf.convert_to_tensor((X.T / whitening_factor).T, dtype=tf.float64)
    y_tensor = tf.reshape(tf.convert_to_tensor(y / whitening_factor,
                                               dtype=tf.float64), (-1, 1))
    eps = tf.convert_to_tensor(eps, dtype=tf.float64)
    one = tf.constant(1., dtype=tf.float64)

    with session or tf.Session() as session:
        # solves the L2 norm with L2 regularization problem
        # and use its solution as initial value for the MM algorithm
        x = tf.matrix_solve_ls(X_tensor, y_tensor, l2_regularizer=l1_regularizer)
        n = 0
        while n < maxiter:
            reg_factor = tf.norm(x, ord=1)
            l1_factor = tf.maximum(eps,
                                   tf.sqrt(tf.abs(y_tensor - tf.matmul(X_tensor, x))))
            # solve the reweighted least squares problem with L2 regularization
            xo = tf.matrix_solve_ls(X_tensor/l1_factor, y_tensor/l1_factor,
                                    l2_regularizer=l1_regularizer/reg_factor)
            # compute stopping criterion
            rel_err = tf.norm(x - xo, ord=1) / tf.maximum(one, reg_factor)
            # update
            x = xo
            if session.run(rel_err) < rtol:
                break
            n += 1
        res = x.eval()
    return res


def nLasso():
    obs = pd.read_csv('../obs.csv', sep=';', delimiter=None, header='infer')
    obs = obs.dropna(subset=['Air temperature (t2m)', '# lat', 'lon'])
    df = obs.drop_duplicates('# lat')
    stat_lat, stat_lon = df['# lat'].values, df['lon'].values

    cluster_1 = [37, 35, 32, 21, 22, 27, 19, 17, 14, 12, 11, 16, 6, 8, 4, 1, 2]
    cluster_2 = [0, 3, 15, 5, 7, 9, 18, 25, 26, 28, 29, 31, 34, 36, 43, 44, 50, 49, 46]
    cluster_3 = [22, 17, 21, 14, 11, 12, 8, 6, 4]
    cluster_4 = [22, 17, 21, 14, 11, 12, 8, 6, 4, 27, 19, 16, 1, 2]

    FMI_stations = [(stat_lat[i], stat_lon[i]) for i in range(len(stat_lat))]
    FMI_stations = sorted(FMI_stations, key=lambda tup: tup[0])
    FMI_stations = np.array(FMI_stations)
    nr_stations = len(FMI_stations)
    N = nr_stations

    # here we create the links between neighboring FMI stations
    K_nn = 3
    Idx = create_links(FMI_stations, K_nn + 1)

    A = np.zeros((nr_stations, nr_stations))
    for iter_obs in range(nr_stations):
        A[iter_obs, Idx[iter_obs]] = 1

    A = A + np.transpose(A)
    A[A > 0.1] = 1
    A = A - np.eye(A.shape[0])

    # number of features at each data point
    d = 3

    X_mtx = np.zeros((d, N))
    true_y = np.zeros(N)

    for iter_station in range(nr_stations):
        # finding near stations
        near_stations = obs.loc[(obs['# lat'] - FMI_stations[iter_station][0]) ** 2 + (
                    obs['lon'] - FMI_stations[iter_station][1]) ** 2 < 0.0001]
        dmy = near_stations['Air temperature (t2m)']
        indices = dmy.dropna()
        indices = indices.sample(frac=1).reset_index(drop=True)

        blocklen = math.floor(len(indices) / (d + 1))

        for iter_dim in range(d):
            X_mtx[iter_dim, iter_station] = sum(
                indices[(iter_dim * blocklen):(blocklen * (iter_dim + 1))].values) / blocklen
        true_y[iter_station] = sum(indices[(d * blocklen):(blocklen * (d + 1))].values) / blocklen

    # create weighted incidence matrix
    G = nx.DiGraph(np.triu(A, 1))
    D = np.transpose(nx.incidence_matrix(G, oriented=True).toarray())
    D_block = np.kron(D, np.eye(d))

    [nr_edges, N] = D.shape

    Lambda = np.diag((1. / (np.sum(abs(D), 1))))
    Lambda_block = np.kron(Lambda, np.eye(d))
    Gamma_vec = (1. / (sum(abs(D))))
    Gamma = np.diag(Gamma_vec)
    Gamma_block = np.kron(Gamma, np.eye(d))

    # Algorithm Initialisation

    _lambda = 1 / 10
    _lambda = 1 / 9
    _lambda = 1 / 7

    RUNS = 1
    MSE = np.zeros(RUNS)
    for iter_runs in range(RUNS):
        dmy_idx = np.ones(N)
        dmy_idx[cluster_3] = 0
        dmy_idx[[11, 12, 14]] = 1
        samplingset = np.where(dmy_idx > 0.2)[0]
        unlabeled = np.where(dmy_idx < 0.2)[0]

        hatx = np.zeros(N * d)
        running_average = np.zeros(N * d)
        haty = np.zeros(nr_edges * d)

        mtx_A_block = np.zeros((N * d, N * d))
        mtx_B_block = np.zeros((N * d, N * d))

        for iter_node in range(N):
            msk_dmy = np.zeros((N, N))
            msk_dmy[iter_node, iter_node] = 1
            tilde_tau = len(samplingset) / (2 * Gamma_vec[iter_node])

            iter_node_vec = X_mtx[:, iter_node]
            mtx_B = np.linalg.inv(np.outer(iter_node_vec.ravel(), iter_node_vec.ravel()) + tilde_tau * np.eye(d))

            mtx_A = np.dot(tilde_tau * np.eye(d), mtx_B)

            mtx_A_block = mtx_A_block + np.kron(msk_dmy, mtx_A)
            mtx_B_block = mtx_B_block + np.kron(msk_dmy, mtx_B)

        # TODO fix true_y
        _mtx = np.dot(X_mtx, np.diag(true_y))
        _mtx = np.reshape(_mtx, (N * d, 1), order="F")
        vec_B = np.dot(mtx_B_block, _mtx).ravel()

        for iterk in range(1000):

            newx = hatx - 0.5 * Gamma_block.dot(np.transpose(D_block).dot(haty))
            newx = update_x_linreg(newx, samplingset, d, N, mtx_A_block, vec_B)

            tildex = 2 * newx - hatx
            newy = haty + (0.5 * Lambda_block).dot(D_block.dot(tildex))
            haty = block_clipping(newy, d, nr_edges, _lambda)
            hatx = newx
            running_average = (running_average * iterk + hatx) / (iterk + 1)

        w_hat = np.reshape(running_average, (d, N), order="F")
        y_hat = sum(np.multiply(X_mtx, w_hat))

        MSE[iter_runs] = np.linalg.norm(y_hat[unlabeled] - true_y[unlabeled], 2) ** 2 / np.linalg.norm(true_y[unlabeled],
                                                                                                       2) ** 2

    A_mtx = X_mtx[:, unlabeled]
    A_mtx = np.transpose(A_mtx)
    y = true_y[unlabeled]
    y = np.transpose(y)
    x = lad(A_mtx, y)

    return np.linalg.norm(np.dot(A_mtx, x).ravel() - y, 2) ** 2 / np.linalg.norm(y, 2) ** 2

