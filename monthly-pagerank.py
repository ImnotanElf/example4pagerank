import time, glob
import networkx as nx

def datetime2timestamp( datetime_str ): # datetime_str = '2021-06-03 21:19:03'
    timeArray = time.strptime( datetime_str, "%Y-%m-%d %H:%M:%S" )
    timeStamp = int( time.mktime( timeArray ) )
    return timeStamp

def timestamp2datetime( timestamp ): # unit: second
    time_local = time.localtime( timestamp )
    dt = time.strftime( "%Y-%m-%d %H:%M:%S", time_local )
    return dt

def get_stamps( every_months = 1 ):
    months = []  
    for x in range( 2023, 2016, -1 ):
        for y in range( 12, 0, -1 ):
            months.append( f"{ x }-{ str( y ).rjust( 2, '0' ) }-01 00:00:00" )
    stamps = [ datetime2timestamp( x ) for x in months ]
    nowstamp = time.time()
    month_index = 0
    for i in range( len( stamps ) ):
        if stamps[ i ] < nowstamp:
            month_index = i
            break
    res = []
    while ( month_index < len( stamps ) ):
        res.append( stamps[ month_index : month_index + every_months + 1 ] )
        month_index += every_months
    res[ 0 ][ 0 ] = nowstamp
    if ( len( res[ -1 ] ) == 1 ):
        res.pop()
    res = [ [ x[ 0 ], x[ -1 ] ] for x in res ]
    return res

def main():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个CSV文件' % len( txt_list ) )
    print( u'正在处理............' )
    map_addr_id = {}
    with open( 'mapped.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ]
            id = int( info[ 1 ] )
            map_addr_id[ addr ] = id
    stamps = get_stamps()
    txs = []
    with open( 'all-txs.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr_from = info[ 0 ].lower()
            addr_to = info[ 1 ].lower()
            tx_stamp = int( info[ 4 ] )
            txs.append( [ addr_from, addr_to, tx_stamp ] )
    for st in stamps:
        startstamp = st[ 0 ]
        endstamp   = st[ 1 ]
        G1 = nx.Graph()
        rate = 0
        for i in txt_list: #循环读取同文件夹下的txt文件
            with open( i, 'r' ) as fr:
                lines = fr.readlines()
                rate += len( lines )
                print( "%.2f%%" % ( rate / 4152464.91 ) )
                for line in lines:
                    info = line.split( ',' )
                    G1.add_edge( map_addr_id[ info[ 0 ].lower() ], map_addr_id[ info[ 1 ].lower() ], weight = 1 )
                    # G1.add_edge( info[ 0 ].lower(), info[ 1 ].lower(), weight = int( info[ 2 ] ) )
        print( "Number of nodes:", len( G1.nodes() ) )

        pr = nx.pagerank(G1, alpha=0.85, personalization=None,
                    max_iter=100000, tol=1.0e-6, nstart=None, weight='weight',
                    dangling=None)
        sorted_pr = sorted( pr.items(), key = lambda x : x[ 1 ] )
        for i in range( 10 ):
            print( sorted_pr[-1 - i] )

if __name__ == "__main__":
    a = timestamp2datetime( 1587983548 )
    a = timestamp2datetime( time.time() )
    print( a )
    b = get_stamps( 3 )
    print( b )