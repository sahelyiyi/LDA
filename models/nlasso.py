import requests
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import networkx as nx
import re
import argparse
import math
import scipy.optimize

from lad import lad
from sklearn.neighbors import NearestNeighbors


def create_links(X_train, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_train)
    _, indices = nbrs.kneighbors(X_train)
    return indices


def block_clipping(weights_in, feature_dim, nr_edges, _lambda):
    mtx = weights_in.reshape(feature_dim, nr_edges)

    x_norm = np.sqrt(sum(np.power(mtx, 2)))

    idx_exceed = np.where(x_norm < _lambda)[0]
    factor = np.ones(nr_edges)

    for iter_idx in range(len(idx_exceed)):
        idx = idx_exceed[iter_idx]
        factor[idx] = np.divide(_lambda, x_norm[idx])

    weights_out = np.dot(mtx, np.diag(factor))
    weights_out = weights_out.ravel()

    return weights_out


def update_x_linreg(weights_in, sampling_set, feature_dim, nr_nodes, mtx_A, vec_B):
    tmp = np.dot(mtx_A, weights_in) + vec_B
    tmp = tmp.reshape(feature_dim, nr_nodes)

    weights_out = weights_in.reshape(feature_dim, nr_nodes)
    weights_out[:, sampling_set] = tmp[:, sampling_set]
    weights_out = weights_out.reshape(feature_dim * nr_nodes)

    return weights_out


def lad(X, y, yerr=None, l1_regularizer=0.5, maxiter=50, rtol=1e-4,
        eps=1e-4, session=None):
    """
    Linear least absolute deviations with L1 norm regularization using
    Majorization-Minimization. See [1]_ for a similar mathematical derivation.

    Parameters
    ----------
    X : (n, m)-matrix
        Design matrix
    y : (n, 1) matrix
        Vector of observations
    yerr : (n, 1) matrix
        Vector of standard deviations on the observations
    l1_regularizer : float
        Factor to control the importance of the L1 regularization.
        Note: due to a limitation of tensorflow.matrix_solve_ls,
        do not set this value to zero
    maxiter : int
        Maximum number of iterations of the majorization-minimization algorithm.
        If maxiter equals zero, then this function returns the Weighted
        Least-Squares coefficients
    rtol : float
        Relative tolerance used as an early stopping criterion
    eps : float
        Inscrease this value if tensorflow raises an exception
        saying that the Cholesky decomposition was not successful
    session : tf.Session object
        A tensorflow.Session object

    Returns
    -------
    x : (m, 1) matrix
        Vector of coefficients that minimizes the least absolute deviations
        with L1 regularization.

    References
    ----------
    [1] Phillips, R. F. Least absolute deviations estimation via the EM
        algorithm. Statistics and Computing, 12, 281-285, 2002.
    """

    if yerr is not None:
        whitening_factor = yerr/math.sqrt(2.)
    else:
        whitening_factor = 1.

    # converts inputs to tensors
    X_tensor = tf.convert_to_tensor((X.T / whitening_factor).T, dtype=tf.float64)
    y_tensor = tf.reshape(tf.convert_to_tensor(y / whitening_factor,
                                               dtype=tf.float64), (-1, 1))
    eps = tf.convert_to_tensor(eps, dtype=tf.float64)

    with session or tf.Session() as session:
        # solves the L2 norm with L2 regularization problem
        # and use its solution as initial value for the MM algorithm
        x = tf.matrix_solve_ls(X_tensor, y_tensor, l2_regularizer=l1_regularizer)
        n = 0
        while n < maxiter:
            reg_factor = tf.norm(x, ord=1)
            l1_factor = tf.maximum(eps, tf.sqrt(tf.abs(y_tensor - tf.matmul(X_tensor, x))))

            X_tensor = X_tensor / l1_factor
            y_tensor = y_tensor / l1_factor

            # Solves the reweighted least squares problem with L2 regularization
            xo = tf.matrix_solve_ls(X_tensor, y_tensor,
                                    l2_regularizer=l1_regularizer/reg_factor)

            rel_err = tf.norm(x - xo, ord=1) / tf.maximum(tf.constant(1., dtype=tf.float64), reg_factor)
            x = xo
            if session.run(rel_err) < rtol:
                break
            n += 1
        res = x.eval()
    return res


def nLasso():
    # TODO fix path ../obs.csv
    obs = pd.read_csv('obs.csv', sep=';', delimiter=None, header='infer')
    obs = obs.dropna(subset=['Air temperature (t2m)', '# lat', 'lon'])
    df = obs.drop_duplicates('# lat')
    stat_lat, stat_lon = df['# lat'].values, df['lon'].values

    cluster_1 = [38, 36, 33, 22, 23, 28, 20, 18, 15, 13, 12, 17, 7, 9, 5, 2, 3]
    cluster_2 = [1, 4, 16, 6, 8, 10, 19, 26, 27, 29, 30, 32, 35, 37, 44, 45, 51, 50, 47]
    cluster_3 = [23, 18, 22, 15, 12, 13, 9, 7, 5]
    cluster_4 = [23, 18, 22, 15, 12, 13, 9, 7, 5, 28, 20, 17, 2, 3]

    FMI_stations = [(stat_lat[i], stat_lon[i]) for i in range(len(stat_lat))]
    nr_stations = len(stat_lat)
    N = nr_stations

    # here we create the links between neighboring FMI stations
    K_nn = 3
    Idx = create_links(FMI_stations, K_nn + 1)

    A = np.zeros((nr_stations, nr_stations))
    for iter_obs in range(nr_stations):
        A[iter_obs, Idx[iter_obs]] = 1

    # A(23,20)=1
    # A(3,12)=1
    # A(17,12)=1
    # A(3,7)=1
    A = A + np.transpose(A)
    A[A > 0.1] = 1
    A = A - np.eye(A.shape[0])

    # G = graph(A,'upper');

    # indicate FMI weather stations by circles
    # geoplot(FMI_stations(:,1),FMI_stations(:,2),'o','MarkerSize',10,'LineWidth',2);
    # indicate Helsinki with red cross
    # geoplot(60.192,24.9458,'rx','MarkerSize',20,'LineWidth',5);

    # measurements = zeros(1,nr_stations);
    # for iter_station=1:nr_stations:
    #    label = sprintf('%3d',iter_station) ;
    #    text(FMI_stations(iter_station,1),FMI_stations(iter_station,2),label,'FontSize',20);

    # number of features at each data point
    d = 3

    X_mtx = np.zeros((d, N))
    # true_y = np.zeros((1, N))
    true_y = np.zeros(N)

    for iter_station in range(nr_stations):
        # finding near stations
        near_stations = obs.loc[(obs['# lat'] - FMI_stations[iter_station][0]) ** 2 + (
                    obs['lon'] - FMI_stations[iter_station][1]) ** 2 < 0.0001]
        dmy = near_stations['Air temperature (t2m)']
        indices = dmy.dropna()

        blocklen = math.floor(len(indices) / (d + 1))

        for iter_dim in range(d):
            X_mtx[iter_dim, iter_station] = sum(
                indices[(iter_dim * blocklen):(blocklen * (iter_dim + 1))].values) / blocklen
        true_y[iter_station] = sum(indices[(d * blocklen):(blocklen * (d + 1))].values) / blocklen

        # idxs = find(A(iter_station,:) > 0.1);
        # draw links between each FMI station and its d nearest neighbours(NN)
        # for iter_inner=1:length(idxs):
        #   geoplot(FMI_stations([iter_station, idxs(iter_inner)], 1), FMI_stations([iter_station, idxs(iter_inner)], 2),
        #          'k-', 'LineWidth', 1)

    nr_nodes = N
    feature_dim = d

    feature_mtx = X_mtx

    mask_mtx = np.kron(np.eye(nr_nodes), np.ones((feature_dim, feature_dim)))
    feature_norm_squared_vec = np.sum(np.power(feature_mtx, 2), axis=0)
    norm_features_2 = np.kron(np.diag(feature_norm_squared_vec), np.eye(feature_dim))
    feature_mtx_vec = np.reshape(feature_mtx, nr_nodes * feature_dim, 1)
    proj_feature = np.dot(np.multiply(np.outer(feature_mtx_vec.ravel(), feature_mtx_vec.ravel()), mask_mtx), np.linalg.inv(
        norm_features_2))

    out_of_proj = np.eye(nr_nodes * feature_dim) - proj_feature

    Adjac = np.triu(A, 1)
    A_undirected = Adjac + np.transpose(Adjac)
    degrees = sum(A_undirected)
    inv_degrees = 1. / degrees

    # create weighted incidence matrix

    G = nx.DiGraph(np.triu(A, 1))
    D = np.transpose(nx.incidence_matrix(G, oriented=True).toarray())
    D_block = np.kron(D, np.eye(d))

    [nr_edges, N] = D.shape

    # edge_weights = zeros(nr_edges,1);
    # for iter_edge=1:M
    #     [s,t] = findedge(G,iter_edge); %finds the source and target nodes of the edges specified by idx.
    #      edge_weights(iter_edge) = sqrt(A_undirected(s,t)) ;
    # D = diag(edge_weights)*D ;

    # some vision
    # scatter(nodes(:,1),nodes(:,2)) ;
    # figure(1);
    # plot(G);

    Lambda = np.diag((1. / (np.sum(abs(D), 1))))
    Lambda_block = np.kron(Lambda, np.eye(d))
    Gamma_vec = (1. / (sum(abs(D))))
    Gamma = np.diag(Gamma_vec)
    Gamma_block = np.kron(Gamma, np.eye(d))

    # Algorithm Initialisation

    # primSLP = np.ones(N)
    # primSLP[N - 1] = 0
    # running_average;
    # dualSLP = np.array([i / N - 1 for i in range(1, N)])
    # dualSLP = np.zeros(nr_edges)

    _lambda = 1 / 10
    _lambda = 1 / 9
    _lambda = 1 / 7

    RUNS = 1
    MSE = np.zeros(RUNS)
    for iter_runs in range(RUNS):
        dmy_idx = np.ones(N)
        dmy_idx[cluster_3] = 0
        dmy_idx[[12, 13, 15]] = 1
        samplingset = np.where(dmy_idx > 0.2)[0]
        unlabeled = np.where(dmy_idx < 0.2)[0]

        hatx = np.zeros(N * d)
        running_average = np.zeros(N * d)
        # haty = np.array((1:(N-1)) / (N - 1));
        haty = np.zeros(nr_edges * d)
        running_averagey = np.zeros(nr_edges * d)
        hatxLP = np.zeros(N * d)
        hatw = np.zeros(d * N)

        # log_conv = zeros(N, 1);
        # log_bound = zeros(N, 1);

        dmy = len(Gamma_vec)
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

        # tilde_tau = length(samplingset) * (1. / (2 * diag(Gamma_block)));
        # mtx_A = diag(ones(dmy, 1). / (ones(dmy, 1) + 2 * Gamma_vec / length(samplingset)));

        # mtx_B = diag(ones(dmy, 1). / (ones(dmy, 1) + tilde_tau));

        # TODO fix true_y
        vec_B = np.dot(mtx_B_block, np.dot(X_mtx, np.diag(true_y)).reshape(N * d))

        for iterk in range(1000):
            # LP iteration
            # hatxLP = inv_degrees. * (A_undirected * hatxLP);
            # hatxLP(samplingset) = graphsig(samplingset);

            newx = hatx - 0.5 * Gamma_block.dot(np.transpose(D_block).dot(haty))

            # SLP iteration
            # newx = block_thresholding(newx, samplingset, y, X_mtx, d, N, Gamma_vec, out_of_proj,
            #                            feature_norm_squared_vec);

            #  update for least squared linear regression
            # TODO fix update_x_linreg
            newx = update_x_linreg(newx, samplingset, d, N, mtx_A_block, vec_B)
            # newx(samplingset) = graphsig(samplingset);

            tildex = 2 * newx - hatx
            newy = haty + (0.5 * Lambda_block).dot(D_block.dot(tildex))
            haty = block_clipping(newy, d, nr_edges, _lambda)
            hatx = newx
            running_average = (running_average * iterk + hatx) / (iterk + 1)

            # running_averagey = (running_average * (iterk - 1) + haty) / iterk;
            # dual = sign(D * running_average);
            # dual(iterk:(N - 1)) = 0;
            # log_conv(iterk) = sum(abs(D * running_average));
            # log_bound(iterk) = (1 / (2 * iterk)) * (primSLP'*inv(Gamma)*primSLP)+((dualSLP-dual)' * inv(Lambda) * (
            #            dualSLP - dual));

        # figure(1);
        # stem(primSLP);
        # title('primal SLP')
        # figure(2);
        # stem(dualSLP);
        # title('dual SLP')
        # figure(2);

        w_hat = running_average.reshape(d, N)

        # norm_w_hat=sum(np.power(w_hat, 2),1);

        y_hat = sum(np.multiply(X_mtx, w_hat))
        # stem([y_hat' true_y']);

        MSE[iter_runs] = np.linalg.norm(y_hat[unlabeled] - true_y[unlabeled], 2) ** 2 / np.linalg.norm(true_y[unlabeled],
                                                                                                       2) ** 2

    # figure(3);
    mtx = np.transpose(running_average.reshape(d, N))
    # stem(mtx(unlabeled,:));
    A_mtx = feature_mtx[:, unlabeled]
    A_mtx = np.transpose(A_mtx)
    y = true_y[unlabeled]
    y = np.transpose(y)
    x = lad(A_mtx, y)
    np.linalg.norm(np.dot(A_mtx, x) - y, 2) ** 2 / np.linalg.norm(y, 2) ** 2

    # T = array2table(mtx,'VariableNames',{'a','b'});
    # filename = sprintf('MSE_FMI_%s.csv',date) ;
    # writetable(T,fullfile(pathtothismfile,filename));

    # figure(4);
    # stem(dual);
    # bound =log_bound+(1./(2*(1:K))*(hatx'*inv(Lambda)*hatx) ;
    # plot(1:N,log([log_conv log_bound]));
