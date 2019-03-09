import numpy as np
import networkx as nx
from tqdm import *

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


class RandomWalkSampler:
    """
    Implementation of random walk method.
    """
    def __init__(self, graph, walk_length, negative_num, batch_size):
        self.node_num = graph.number_of_nodes()
        self.node_index = 0
        self.walk_length = walk_length
        self.negative_num = negative_num
        self.batch_size = batch_size
        
        # Calculate transformation probability for every node,
        # then use that probability distribution to build Alias sampling model.
        adj_matrix = nx.adj_matrix(graph).toarray()
        self.samplings = []
        with tqdm(total=self.node_num, desc='Building sampling model') as pbar:
            for i in range(self.node_num):
                node_weights = adj_matrix[i]
                weight_distribution = node_weights / np.sum(node_weights)
                self.samplings.append(AliasSampling(weight_distribution))
                pbar.update(1)
        
        node_degrees = np.sum(adj_matrix, axis=0)
        node_distribution = node_degrees / np.sum(node_degrees)
        self.node_sampling = AliasSampling(node_distribution)
        
    def get_one_sample(self):
        positive_nodes = self.walk_on_node(self.node_index)
        negative_nodes = self._generate_negative([self.node_index] + positive_nodes, self.negative_num)
        u_i = np.array([self.node_index] * (self.walk_length + self.negative_num))
        u_j = np.array(positive_nodes + negative_nodes)
        label = np.array([1] * self.walk_length + [-1] * self.negative_num)
        self.node_index = (self.node_index + 1) % self.node_num
        return u_i, u_j, label
    
    def get_one_batch(self):
        u_is, u_js, labels = [], [], []
        for i in range(self.batch_size):
            u_i, u_j, label = self.get_one_sample()
            u_is.append(u_i)
            u_js.append(u_j)
            labels.append(label)
        return np.concatenate(u_is, axis=None), np.concatenate(u_js, axis=None), np.concatenate(labels, axis=None)
    
    def walk_on_node(self, node):
        result = []
        current_node = node
        for i in range(self.walk_length):
            current_node = self._walk_one_step(current_node)
            result.append(current_node)
        return result
            
    def _walk_one_step(self, start_node):
        """
        Walk just one step from start node. Returns the next node.
        """
        return self.samplings[start_node].sampling()
    
    def _generate_negative(self, positive_samples, negative_num):
        negative_result = []
        while (len(negative_result) < negative_num):
            neg_sample = self.node_sampling.sampling()
            if neg_sample in positive_samples:
                continue
            negative_result.append(neg_sample)
        return negative_result


class EdgeSampler:
    def __init__(self, graph, batch_size, K):
        self.graph = graph
        self.batch_size = batch_size
        self.K = K
        
        self.number_of_nodes = self.graph.number_of_nodes()
        self.number_of_edges = self.graph.number_of_edges()
        
        self.edges_raw = self.graph.edges(data=True)
        self.nodes_raw = self.graph.nodes(data=True)
        
        weight_distribution = np.array([edge[2]['weight'] for edge in self.edges_raw]) + 0.0
        weight_distribution /= np.sum(weight_distribution)
        self.edge_sampling = AliasSampling(prob=weight_distribution)
        
        node_negative_distribution = np.power(
            np.array([self.graph.degree(node, weight='weight') 
                      for node, attr in self.nodes_raw], dtype=np.float32), 0.75)
        node_negative_distribution /= np.sum(node_negative_distribution)
        self.node_sampling = AliasSampling(prob=node_negative_distribution)
        
        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, attr) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, attr in self.edges_raw]
        print("Finished edge sampler initialization.")
    
    def next_batch(self):
        """
        Fetch next batch of edges.
        
        @params K: number of negative samples.
        """
        # Sample from the original edges and treat them as binary edges, 
        # with the sampling probabilities proportional to the original edge weights.
        edge_batch_index = self.edge_sampling.sampling(self.batch_size)
        u_i, u_j, label = [], [], []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            
            for i in range(self.K):
                while True:
                    negative_node = self.node_sampling.sampling()
                    # Keep randomly select nodes util there is no edge between current node and selected node.
                    if not self.graph.has_edge(self.node_index_reversed[negative_node], 
                                               self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, attr in self.nodes_raw}