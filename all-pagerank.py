import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import glob

def main():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个CSV文件' % len( txt_list ) )
    print( u'正在处理............' )
    G1 = nx.Graph()
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            print( len( lines ) )
            for line in lines:
                info = line.split( ',' )
                G1.add_edge( info[ 0 ].lower(), info[ 1 ].lower(), weight = 1 )
                # G1.add_edge( info[ 0 ].lower(), info[ 1 ].lower(), weight = int( info[ 2 ] ) )
    print( "Number of nodes:", len( G1.nodes() ) )

    pr = nx.pagerank(G1, alpha=0.85, personalization=None,
                max_iter=100000, tol=1.0e-6, nstart=None, weight='weight',
                dangling=None)
    sorted_pr = sorted( pr.items(), key = lambda x : x[ 1 ] )
    for i in range( 10 ):
        print( sorted_pr[-1 - i] )

if __name__ == "__main__":
    main()