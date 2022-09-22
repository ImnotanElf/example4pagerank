import networkx as nx

def test():
    G = nx.DiGraph()
    G.add_edge( 1, 2 )
    pr = nx.pagerank( G, alpha=0.85, personalization=None,
                max_iter=100000, tol=1.0e-6, nstart=None, weight=None,
                dangling=None)
    print( pr )

if __name__ == "__main__":
    test()