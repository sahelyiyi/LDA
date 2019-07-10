from __future__ import print_function
from math import *
import time

import numpy as np
from scipy.special import gammaln, psi
from scipy.stats import norm
from six.moves import xrange

from settings import EPS, RTM_RESULTS_FILE_PATH, RTM_ALPHA, RTM_RHO


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RTM:
    #  implementation of relational topic models by Chang and Blei (2009)

    def __init__(self, number_topics, number_docs, number_vocab, link_function='exponential', alpha=RTM_ALPHA,
                 rho=RTM_RHO, **kwargs):
        self.number_topics = number_topics
        self.number_docs = number_docs
        self.number_vocab = number_vocab

        if link_function in ['sigmoid', 'exponential', 'phi', 'N']:
            self.link_function = link_function
        else:
            print('Invalid link function found.')
            return

        self.alpha = alpha

        # gamma is per-document topic distributions
        self.gamma = np.random.gamma(100., 1. / 100, [self.number_docs, self.number_topics])
        # beta is per-topic word distributions
        self.beta = np.random.dirichlet([5] * self.number_vocab, self.number_topics)

        self.v = 0
        self.eta = np.random.normal(0., 1, self.number_topics)

        self.phi = list()
        self.pi = np.zeros([self.number_docs, self.number_topics])

        self.rho = rho

        self.verbose = kwargs.pop('verbose', True)

    def fit(self, paper_words_count_matrix, doc_links, max_iter=100):
        for doc in xrange(self.number_docs):

            doc_data = paper_words_count_matrix.getrow(doc)
            unique_words = len(doc_data.indices)
            unique_words_cnt = doc_data.data

            self.phi.append(np.random.dirichlet([10] * self.number_topics,
                                                unique_words).T)  # list of KxW, document words distributions for topics

            self.pi[doc, :] = np.sum(unique_words_cnt * self.phi[doc], 1) / np.sum(
                unique_words_cnt * self.phi[doc])  # document topic distribution

        for iter in xrange(max_iter):
            tic = time.time()

            self.inference(paper_words_count_matrix, doc_links)

            self.parameter_estimation(doc_links)

            if self.verbose:
                elbo = self.compute_ELBO(paper_words_count_matrix, doc_links)
                print('ITER %d:\ttime: %f, ELBO: %f' % (iter, time.time() - tic, elbo))

    def inference(self, paper_words_count_matrix, doc_links):
        # update phi, gamma, Appendix A
        E_log_theta_given_gamma = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]

        new_beta = np.zeros([self.number_topics, self.number_vocab])  # TODO check dimentions

        for d1 in xrange(self.number_docs):
            doc_data = paper_words_count_matrix.getrow(d1)
            words = doc_data.indices
            words_cnt = doc_data.data
            doc_len = np.sum(words_cnt)

            gradient = np.zeros(self.number_topics)
            for d2 in doc_links[d1]:
                var = np.dot(self.eta, self.pi[d1] * self.pi[d2]) + self.v
                if self.link_function == 'sigmoid':
                    coefficient = (1 - sigmoid(var)) * self.eta
                elif self.link_function == 'exponential':
                    coefficient = self.eta
                elif self.link_function == 'phi':
                    coefficient = (norm.pdf(var) / norm.cdf(var)) * self.eta
                else:
                    coefficient = self.eta  # TODO implement for N method
                gradient += coefficient * self.pi[d2, :] / doc_len

            new_phi = np.exp(gradient[:, np.newaxis] + E_log_theta_given_gamma[d1, :][:, np.newaxis] + np.log(
                self.beta[:, words] + EPS))
            new_phi = new_phi / np.sum(new_phi, 0)

            self.phi[d1] = new_phi

            self.pi[d1, :] = np.sum(words_cnt * self.phi[d1], 1) / np.sum(words_cnt * self.phi[d1])
            self.gamma[d1, :] = self.alpha + np.sum(words_cnt * self.phi[d1], 1)
            new_beta[:, words] += (words_cnt * self.phi[d1])

        self.beta = new_beta / np.sum(new_beta, 1)[:, np.newaxis]

    def parameter_estimation(self, doc_links):
        # update eta, v, Appendix B
        large_pi = np.zeros(self.number_topics)
        M = 0.0

        for d1 in xrange(self.number_docs):
            for d2 in doc_links[d1]:
                if d2 < d1:
                    continue
                large_pi += self.pi[d1, :] * self.pi[d2, :]
                M += 1

        pi_alpha = (np.array([self.alpha] * self.number_topics) / (self.alpha * self.number_topics)) ** 2

        self.v = np.log(M - np.sum(large_pi) + EPS) - np.log(self.rho * (1 - np.sum(pi_alpha)) + M - np.sum(large_pi) + EPS)
        self.eta = np.log(large_pi + EPS) - np.log(large_pi + self.rho * pi_alpha + EPS) - self.v

    def compute_ELBO(self, paper_words_count_matrix, doc_links):
        # Appendix A
        elbo = 0

        E_log_theta_given_gamma = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]  # D x K
        log_beta = np.log(self.beta + EPS)

        for d in xrange(self.number_docs):
            doc_data = paper_words_count_matrix.getrow(d)
            words = doc_data.indices
            cnt = doc_data.data

            l_d1_d2 = 0.0
            for d2 in doc_links[d]:
                var = np.dot(self.eta, self.pi[d] * self.pi[d2]) + self.v
                if self.link_function == 'sigmoid':
                    value = log(sigmoid(var))
                elif self.link_function == 'exponential':
                    value = var
                elif self.link_function == 'phi':
                    value = log(norm.cdf(var))
                else:
                    value = var  # TODO implement for N method
                l_d1_d2 += value

            elbo += l_d1_d2
            elbo += np.sum(cnt * (self.phi[d] * log_beta[:, words]))
            elbo += np.sum(self.phi[d].T * E_log_theta_given_gamma[d, :])
            elbo += np.sum((self.alpha - 1.) * E_log_theta_given_gamma[d, :])
            elbo += np.sum(cnt * self.phi[d] * np.log(self.phi[d]) + EPS)
            elbo += -np.sum((self.gamma[d, :] - 1.) * (E_log_theta_given_gamma[d, :]))
            elbo += np.sum(gammaln(self.gamma[d, :])) - gammaln(np.sum(self.gamma[d, :]))

        return elbo

    def save_model(self):
        np.savetxt(RTM_RESULTS_FILE_PATH + '/eta.txt', self.eta, delimiter='\t')
        np.savetxt(RTM_RESULTS_FILE_PATH + '/beta.txt', self.beta, delimiter='\t')
        np.savetxt(RTM_RESULTS_FILE_PATH + '/gamma.txt', self.gamma, delimiter='\t')
        with open(RTM_RESULTS_FILE_PATH + '/v.txt', 'w') as f:
            f.write('%f\n' % self.v)


def print_rtm_topics(model, words, number_topics, number_words):
    print("Topics found via RTM:")
    for topic in range(number_topics):
        top_words = words[model.beta[topic].argsort()[::-1][:number_words]]
        print("\nTopic #%d:" % topic)
        print(" ".join(top_words))
