class GraRep:
    """
    This is the base class of graph representation network.
    """
    def __init__(self, graph, node_features=None, embed_dim=16, batch_size=8, learning_rate=1e-4, regularization=0.1):
        self.graph = graph
        self.node_num = self.graph.number_of_nodes()
        self.edge_num = self.graph.number_of_edges()

        if node_features is not None:
            self.node_features = node_features
            self.feature_dim = self.node_features.shape[1]
        
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization

    def _init_params(self):
        pass

    def _construct_network(self):
        pass