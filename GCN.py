import os
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import networkx as nx
from tqdm import *
import matplotlib.pyplot as plt

from GraRep import GraRep
from Sampling import EdgeSampler


def cal_lapiacian_matrix(A):
    """
    Calculate the given adjacency matrix's 
    symmetric lapiacian matrix $D^{-1/2}LD^{-1/2}.$
    """
    I = np.diag(np.ones(A.shape[0], dtype=np.float32))
    D_diag = A.sum(axis=1)
    D_ = np.diag(np.power(D_diag, -1/2))
    return I - np.matmul(np.matmul(D_, A), D_)


def cal_gcn_matrix(A):
    """
    Calculate the matrix GCN used as a 
    preprocessing weight for graph signal matrix.
    """
    I = np.diag(np.ones(A.shape[0], dtype=np.float32))
    D_diag = A.sum(axis=1)
    D_ = np.diag(np.power(D_diag, -1/2))
    D_[D_ == np.inf] = 0
    
    return I + np.matmul(np.matmul(D_, A), D_)


def cal_poi_stat(poi_matrix):
    statistics = []
    for point in poi_matrix:
        type_statistic = {'05': 0, '07': 0, '12': 0}
        for poi in point:
            typecode_prefix = poi['typecode'][:2]
            try:
                type_statistic[typecode_prefix] += 1
            except KeyError:
                pass
        statistics.append(type_statistic)
    stat_df = pd.DataFrame(statistics)
    stat_df['sum'] = stat_df[['05', '07', '12']].sum(axis=1)
    stat_df['05'] /= stat_df['sum']
    stat_df['07'] /= stat_df['sum']
    stat_df['12'] /= stat_df['sum']
    return stat_df


# Define network architecture
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float64)

class GCN(GraRep):
    def __init__(self, graph, node_features, first_layer_dim=16, embed_dim=16, neg_num=5, batch_size=8):
        super().__init__(graph, node_features, embed_dim=embed_dim, learning_rate=1e-4)
        self.neg_num = neg_num
        self.sample_num = self.batch_size * (1+self.neg_num)
        
        tf.reset_default_graph()
        self.first_layer_dim = first_layer_dim
        
        self._init_params()
        self._construct_network()
        self._optimize_line()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
    def _init_params(self):
        self.w0 = tf.Variable(xavier_init(self.feature_dim, self.first_layer_dim))
        self.b0 = tf.Variable(tf.zeros([self.first_layer_dim], dtype=tf.float64))
        self.w1 = tf.Variable(xavier_init(self.first_layer_dim, self.embed_dim))
        self.b1 = tf.Variable(tf.zeros([self.embed_dim], dtype=tf.float64))
        
        self.adj_matrix = nx.adj_matrix(self.graph).toarray()
        self.gcn_matrix = cal_gcn_matrix(self.adj_matrix)
        
    def _construct_network(self):
        # First layer GCN.
        self.hidden = tf.nn.relu(tf.add(tf.matmul(np.matmul(self.gcn_matrix, self.node_features), self.w0), self.b0))
        # self.hidden = tf.nn.relu(tf.matmul(self.node_features, self.w0))
        # Second layer GCN.
        self.embed = tf.add(tf.matmul(tf.matmul(self.gcn_matrix, self.hidden), self.w1), self.b1)
        # self.embed = tf.matmul(self.hidden, self.w1)

    def _optimize_line(self):
        """
        Unsupervised traininig in LINE manner.
        """
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[self.sample_num])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[self.sample_num])
        self.label = tf.placeholder(name='label', dtype=tf.float64, shape=[self.sample_num])
        
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=self.node_num, 
                                                  dtype=tf.float64), self.embed)
        self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=self.node_num, 
                                                  dtype=tf.float64), self.embed)
        
        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
        self.line_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def train_line(self, u_i, u_j, label):
        """
        Train one minibatch.
        """
        feed_dict = {self.u_i: u_i, self.u_j: u_j, self.label: label}
        _, loss = self.sess.run((self.line_optimizer, self.loss), feed_dict=feed_dict)
        return loss
    
    def cal_embed(self):
        return self.sess.run(self.embed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    with open(os.path.join('data', 'chengdu', 'graph', 'october_trim20.pkl'), 'rb') as file:
        graph = pickle.load(file)

    from sklearn.preprocessing import normalize
    node_attr = normalize(nx.adj_matrix(graph).toarray())

    neg_num = 5
    batch_size = 8
    total_batch = 10000
    display_batch = 100

    model = GCN(graph, node_attr, neg_num=neg_num, batch_size=batch_size)
    sampler = EdgeSampler(graph, batch_size, neg_num)

    avg_loss = 0.
    with trange(total_batch, desc='loss %8.6f' % 0.) as pbar:
        for i in pbar:
            u_i, u_j, label = sampler.next_batch()
            loss = model.train_line(u_i, u_j, label)
            avg_loss += loss / display_batch
            if i % display_batch == 0 and i > 0:
                pbar.set_description('loss %8.6f' % avg_loss)
                avg_loss = 0.

    embed_matrix = model.cal_embed()
    with open(os.path.join('data', 'chengdu', 'embed', 'embed_GCN.pkl'), 'wb') as file:
        pickle.dump(embed_matrix, file)
