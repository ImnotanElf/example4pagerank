import numpy as np
import pandas as pd
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
from torch_geometric import loader
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
import csv
from torch import nn
import time

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, edge_weight):
        x = F.relu(self.gc1(x, adj, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight)
        return F.log_softmax(x, dim=1)

def init_data():
    y = torch.randint(0, 8, [100]).long()   # nodes_labeled
    x = torch.rand([100, 2])    # nodes_features

    rows = np.random.choice(100, 500)     # froms
    cols = np.random.choice(100, 500)     # tos
    rc = np.array( [ rows, cols ] )

    edge_index = torch.tensor( rc ).long()
    edges_weight = torch.rand( [ 100, 3 ] ) # edges_weights
    return x, edge_index, edges_weight, y

'''
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
'''

def main():
    start = time.time()

    x, edge_index, edges_weight, y = init_data()
    graph = data.Data( x = x, edge_index = edge_index, edges_weight = edges_weight, y = y )

    model = GCN( nfeat=graph.x.shape[1], nhid = 128, nclass = graph.y.unique().shape[0] )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    def train( epoch = -1 ):
        optimizer.zero_grad()
        out = model( graph.x, graph.edge_index, graph.edge_weight )
        # print( out.shape )
        loss = F.nll_loss( out, graph.y )
        loss.backward()
        optimizer.step()
        print( 'Epoch: {:04d}'.format( epoch + 1 ), 'loss_train: {:.4f}'.format( loss.item() ) )
        return out

    '''
    @torch.no_grad()
    def test():
        model.eval()
        out = model( graph.x, graph.edge_index, graph.edge_weight )
        # print( out.shape )
        return F.nll_loss( out, graph.y )
    '''

    out = train()
    for epoch in range( 100 ):
        out = train( epoch )
        # test()

    print( out )
    _, pred = out.max( dim = 1 )

    out = F.softmax( out, dim = 1 )
    print( out )


    end = time.time()
    duration = end - start
    print( "Cost time: ", duration )

if __name__ == "__main__":
    main()