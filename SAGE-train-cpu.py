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
import time, glob, argparse, random

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

def init_t_data():
    nnode = 4
    nedge = 5
    nclass = 2
    nfeat  = 0
    y = [ 0, 1, 1, 0 ]
    y = torch.tensor( y ).long()
    x = torch.rand( [ nnode, nfeat] )
    rows = [ 0, 0, 1, 1, 2 ]
    cols = [ 1, 2, 2, 3, 3 ]
    new_rows = np.concatenate( [ rows, cols ] )
    new_cols = np.concatenate( [ cols, rows ] )
    edge_index = torch.tensor( np.array( [new_rows, new_cols] ) ).long()
    edges_weight = torch.rand( 2 * nedge, 1 ).numpy()
    print( x )
    print( edge_index )
    print( type( edges_weight ) )
    print( y )
    return x, edge_index, edges_weight, y , nclass, nfeat

def init_test_data():
    nnode = 4
    nedge = 5
    nclass = 2
    nfeat  = 0
    # y = torch.randint(0, nclass, [ nnode ]).long()
    y = [ _ % nclass for _ in range( nnode ) ]
    y = torch.tensor( y )
    x = torch.rand([ nnode, nfeat])
    rows = np.random.choice( nnode, nedge )
    cols = np.random.choice( nnode, nedge )
    new_rows = np.concatenate( [ rows, cols ] )
    new_cols = np.concatenate( [ cols, rows ] )
    edge_index = torch.tensor( np.array( [new_rows, new_cols] ) ).long()
    edges_weight = torch.rand( 2 * nedge ).numpy()
    print( type( x ) )
    print( type( edge_index ) )
    print( type( edges_weight ) )
    print( type( y ) )
    return x, edge_index, edges_weight, y , nclass, nfeat

def init_data( train_rate ):
    # nnode = 4
    # nedge = 5
    # nclass = 2
    is_one_hot = True
    print( f"Reading train-{ train_rate }-labeled.txt" )
    labeled_dict = {}
    addr_id_dict = {}
    nodes = []
    nodes_label = []
    count = 0
    cate_count = 0
    cate_dict = {}
    with open( f'addr-txs-count-{ train_rate }.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            txs_count = int( info[ 1 ] )
            cate = info[ 2 ].strip()
            if cate not in cate_dict:
                cate_dict[ cate ] = cate_count
                cate_count += 1
            labeled_dict[ addr ] = cate
            addr_id_dict[ addr ] = count
            nodes.append( [ txs_count ] ) 
            nodes_label.append( cate_dict[ cate ] )
            count += 1
    print( f"Read train-{ train_rate }-labeled.txt done" )
    print( "nclass : ", len( cate_dict ) )
    for k, v in cate_dict.items():
        print( k, v )

    path = f"./outs/train-{ train_rate }/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u"./outs/ " + u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    rate = 0
    nodes_from = []
    nodes_to = []
    edge_features = []
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            rate += len( lines )
            # print( "%.2f%%" % ( rate / 308.29 ) )
            for line in lines:
                info = line.strip().split( ',' )
                node_from = info[ 0 ].lower()
                node_to   = info[ 1 ].lower()
                nodes_from.append( addr_id_dict[ node_from ] )
                nodes_to.append( addr_id_dict[ node_to ] )
                edge_features.append( [ int( info[ 2 ] ), int( info[ 3 ] ), int( info[ 4 ] ) ] )
    y = nodes_label
    one_hot_y = encode_onehot( y )
    x = []
    for i in range( len( nodes ) ):
        tt = [ random.random() for _ in range( len( y ) ) ]
        tmp = np.concatenate( [ nodes[ i ], tt ] )
        x.append( tmp )
    # x = nodes

    nfeat = len( x[ 0 ] )
    nnodes = len( x )

    # y = y[ : int( 0.6 * nnodes ) ]

    if nfeat == 0:
        x = [ [] for _ in nodes ]

    train_mask = [ False for _ in range( nnodes ) ]
    for i in range( int ( 0.6 * nnodes ) ):
        train_mask[ i ] = True
    val_mask = [ False for _ in range( nnodes ) ]
    for i in range( int ( 0.6 * nnodes ), int( 0.8 * nnodes ) ):
        train_mask[ i ] = True
    test_mask = [ False for _ in range( nnodes ) ]
    for i in range( int ( 0.8 * nnodes ), nnodes ):
        train_mask[ i ] = True

    rows = nodes_from
    cols = nodes_to
    new_rows = np.concatenate( [ rows, cols ] )
    new_cols = np.concatenate( [ cols, rows ] )
    edge_index = torch.tensor( np.array( [ new_rows, new_cols ] ) ).long()

    edges_weight = np.concatenate( [ edge_features, edge_features ] )
    assert( len( edge_index[ 0 ] ) == len( edges_weight ) )
    return torch.tensor( x, dtype = torch.float32 ), torch.tensor( edge_index ), np.array( edges_weight ), torch.tensor( y ), len( cate_dict ), nfeat, cate_dict, train_mask, val_mask, test_mask, nnodes

def encode_onehot( labels ):
    classes = set( labels )
    classes_dict = { c:np.identity( len( classes ) )[ i, : ] for i, c in enumerate( classes ) }
    labels_onehot = np.array( list( map( classes_dict.get, labels ) ), dtype = np.int32 )
    return labels_onehot

def main( train_rate, nepoch ):
    start = time.time()

    print( "Begin to read data..." )
    x, edge_index, edges_weight, y, nclass, nfeat, cate_dict, train_mask, val_mask, test_mask, nnodes = init_data( train_rate )
    # x, edge_index, edges_weight, y, nclass, nfeat = init_test_data()
    print( "Read data done..." )

    print( "Begin to build Graph..." )
    graph = data.Data( x = x, edge_index = edge_index, edges_weight = edges_weight, y = y, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask )
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
    print( f"Begin to run { nepoch } epoches..." )
    for epoch in range( nepoch ):
        out = train( epoch )
        # test()
    print( f"Run { nepoch } epoches done..." )

    # print( out )
    _, pred = out.max( dim = 1 )
    correct = int( pred[ graph.test_mask ].eq( graph.y[ graph.test_mask ] ).sum().item() )
    print( "accuracy on test: ", correct / ( nnodes * 0.2 ) )

    out = F.softmax( out, dim = 1 )
    # print( pred.tolist() )
    y = y.tolist()
    pred = pred.tolist()
    accuracy = 0
    p_dict = {}
    y_dict = {}
    for i in range( len( y ) ):
        # print( pred[ i ], y[ i ] )
        if pred[ i ] in p_dict:
            p_dict[ pred[ i ] ] += 1
        else:
            p_dict[ pred[ i ] ] = 1
        if y[ i ] in y_dict:
            y_dict[ y[ i ] ] += 1
        else:
            y_dict[ y[ i ] ] = 1
        if ( pred[ i ] == y[ i ] ):
            accuracy += 1
    print( "number of nodes: ", len( y ) )
    print( "prediction:" )
    for k, v in p_dict.items():
        print( k, v )
    print( "real category:" )
    for k, v in y_dict.items():
        print( k, v )
    print( "accuracy rate: {:.2f}%".format( accuracy * 100 / ( len( y ) ) ) )



    end = time.time()
    duration = end - start
    print( "Cost time: ", duration )

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='GCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '--train_rate', type=float, default=0.5, help='train rate' )
    parser.add_argument( '--nepoch', type=int, default=100, help='number of epoches' )
    args = parser.parse_args()

    train_rate = args.train_rate
    nepoch = args.nepoch
    main( train_rate, nepoch )