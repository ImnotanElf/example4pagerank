import numpy as np
import pandas as pd
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
y = torch.randint(0, 8, [100]).long()
node_features = torch.rand([100, 20])
#print(node_features)
rows = np.random.choice(100, 500)
cols = np.random.choice(100, 500)
edges = torch.tensor([rows, cols]).long()
edges_weight = torch.rand(500,1).numpy()
import torch_geometric.data as data
graph = data.Data(x=node_features, edge_index=edges, edges_weight = edges_weight, y=y)
import torch_geometric.data as data
from torch_geometric import loader
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
import csv
from torch import nn

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

node_features= torch.rand((100, 16), dtype=torch.float)
y = torch.randint(0, 8, [100]).long()
y = set(y)
print(y)
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


model = GCN(nfeat=graph.x.shape[1], nhid=128, nclass=graph.y.unique().shape[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()


def train():
  optimizer.zero_grad()
  out = model(graph.x, graph.edge_index, graph.edge_weight)
  loss = F.nll_loss(out, graph.y)
  loss.backward()
  optimizer.step()
  print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss.item()))

@torch.no_grad()
def test():
    model.eval()
    out = model(graph.x, graph.edge_index, graph.edge_weight)
    return F.nll_loss(out, graph.y)

for epoch in range(100):
    train()
    print(test())
    print('*'*100)