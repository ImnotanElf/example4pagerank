#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In this code we try to test the efficiency of two network package in python 3.6: networkx and igraph
import igraph
# import cairo
# import pandas as pd
# import os
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from pylab import *
import time
'''
# os.chdir('/Users/husonghua/Desktop/EV+SUBWAY/')

# Build the edge dataframe.
# The size of this dataframe is large in order to test the ability of this two package.
# That could be some time consuming.
SB_NUM = 5000
SB_NET = pd.DataFrame({'SUB_O': np.repeat(range(1, SB_NUM + 1), SB_NUM),
					   'SUB_D': list(range(1, SB_NUM + 1)) * SB_NUM,
					   'SUB_T': [random.randint(1, 100) for x in range(SB_NUM * SB_NUM)]})
# Drop the O-D have the same O and D.
SB_NET = SB_NET.drop(SB_NET[SB_NET['SUB_O'] == SB_NET['SUB_D']].index)
'''
#################### TEST:igraph ####################
# Change dataframe to tuple; and build the igraph with weight and direction
start_time = time.time()
# If the weight of A--B is not the same as B--A ,then we must be sure to set directed to True.
COM_NET_IG = igraph.Graph()
# COM_NET_IG.to_directed()
COM_NET_IG.add_vertex( 0 )
COM_NET_IG.add_vertex( 1 )
COM_NET_IG.add_vertex( 2 )
COM_NET_IG.add_edge( 0, 1 )
COM_NET_IG.add_edge( 1, 2 )
print( len( COM_NET_IG.get_edgelist() ) )
print( COM_NET_IG.get_vertex_dataframe() )



elapsed_time = time.time() - start_time
print('###  Build the IGRAPH: Cost time:' + str(elapsed_time / 60) + ' min  #####')
# Try to calculate all the shorest path (one O-D) and their length
# It is worth mentioning that in igraph the index is from 0.
# In other words, if our OD index is from 1, then after bulid into igraph, all index should subtract
# So, 1--4 is equal to 2--5 in networkx

start_time = time.time()

pr = COM_NET_IG.pagerank( vertices=None, directed=True, damping = 0.85, weights = None, arpack_options = None, implementation = 'prpack', niter = 100000, eps = 0.000001 )
        
print( len( pr ) )
for i in range( len( pr ) ):
    print( pr[ i ] )
elapsed_time = time.time() - start_time
print('###  Find shorest path in IGRAPH: Cost time:' + str(elapsed_time / 60) + ' min  #####')
#################### TEST:igraph ####################