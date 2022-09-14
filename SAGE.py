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
import time, glob

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

def init_test_data():
    nclass = 8
    nfeat  = 0
    y = torch.randint(0, nclass, [100]).long()
    y = [ _ % 8 for _ in range( 100 ) ]
    y = torch.tensor( y )
    x = torch.rand([100, nfeat])
    print( x )
    #print(node_features)
    rows = np.random.choice(100, 500)
    cols = np.random.choice(100, 500)
    new_rows = np.concatenate( [ rows, cols ] )
    new_cols = np.concatenate( [ cols, rows ] )
    print( ( new_rows.shape ) )
    edge_index = torch.tensor([new_rows, new_cols]).long()
    edges_weight = torch.rand(1000,1).numpy()
    print( type( x ) )
    print( type( edge_index ) )
    print( type( edges_weight ) )
    print( type( y ) )
    return x, edge_index, edges_weight, y , nclass, nfeat

def init_data():
    print( "Reading labeled.txt" )
    labeled_dict = {}
    addr_id_dict = {}
    nodes = []
    nodes_label = []
    count = 0
    cate_count = 0
    cate_dict = {}
    with open( './labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            cate = info[ 1 ]
            if cate not in cate_dict:
                cate_dict[ cate ] = cate_count
                cate_count += 1
            labeled_dict[ addr ] = cate
            addr_id_dict[ addr ] = count
            nodes.append( [ count ] )
            nodes_label.append( cate_dict[ cate ] )
            count += 1
    print( "Read labeled.txt done" )
    print( "nclass : ", len( cate_dict ) )
    

    path = "/data/dev/outs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u"/data/dev/outs/ " + u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    rate = 0
    nodes_from = []
    nodes_to = []
    edge_features = []
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            rate += len( lines )
            print( "%.2f%%" % ( rate / 140912.17 ) )
            for line in lines:
                info = line.strip().split( ',' )
                node_from = info[ 0 ].lower()
                node_to   = info[ 1 ].lower()
                nodes_from.append( addr_id_dict[ node_from ] )
                nodes_to.append( addr_id_dict[ node_to ] )
                edge_features.append( [ int( info[ 2 ] ), int( info[ 3 ] ), int( info[ 4 ] ) ] )
    y = nodes_label
    x = nodes

    rows = nodes_from
    cols = nodes_to
    rc = np.array( [ rows, cols ] )

    edge_index = torch.tensor( rc ).long()
    edges_weight = edge_features
    return torch.tensor( x ).long(), torch.tensor( edge_index ).long(), np.array( edges_weight ), torch.tensor( y ).long(), len( cate_dict ), 0

'''
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
'''

def main():
    start = time.time()

    print( "Begin to read data..." )
    x, edge_index, edges_weight, y, nclass, nfeat = init_test_data()
    print( "Read data done..." )

    print( "Begin to build Graph..." )
    graph = data.Data( x = x, edge_index = edge_index, edges_weight = edges_weight, y = y )
    print( "Graph build done..." )

    print( "Begin to build GCN..." )
    model = GCN( nfeat = nfeat, nhid = 128, nclass = nclass )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print( "GCN build done..." )

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
    print( "Begin to run 100 epoches..." )
    for epoch in range( 100 ):
        out = train( epoch )
        # test()
    print( "Run 100 epoches done..." )

    print( out )
    _, pred = out.max( dim = 1 )

    out = F.softmax( out, dim = 1 )
    print( out )


    end = time.time()
    duration = end - start
    print( "Cost time: ", duration )

if __name__ == "__main__":
    main()