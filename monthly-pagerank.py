import time, glob, argparse
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

def get_addr_id_dict():
    map_addr_id = {}
    with open( 'mapped.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            id = int( info[ 1 ] )
            map_addr_id[ addr ] = id
    return map_addr_id

def process( timestamp ):
    print( "Read map-addr-id..." )
    # map_addr_id = get_addr_id_dict()
    print( "Read map-addr-id done." )

    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )

    G1 = nx.Graph()
    rate = 0
    count_self = 0
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            rate += len( lines )
            info_last  = lines[ -1 ].strip().split( ',' )
            info_first = lines[  0 ].strip().split( ',' )
            if int( info_last[ 4 ] ) < timestamp:
                continue
            print( "{:0.2f}%".format( rate / 11497057.07 ) )
            for line in lines:
                info = line.strip().split( ',' )
                tx_from  = info[ 0 ].lower()
                tx_to    = info[ 1 ].lower()
                tx_value = int( info[ 2 ] )
                tx_blockheight = int( info[ 3 ] )
                tx_timestamp = int( info[ 4 ] )
                if tx_timestamp < timestamp:
                    continue
                if tx_from == '' or tx_to == '':
                    count += 1
                    break
                G1.add_edge( tx_from, tx_to, weight = 1 )
    print( "Number of nodes:", len( G1.nodes() ) )
    print( "count_self: ", count_self )
    # print( "len( map ): ", len( map_addr_id ) )

    pr = nx.pagerank(G1, alpha=0.85, personalization=None,
                max_iter=100000, tol=1.0e-6, nstart=None, weight='weight',
                dangling=None)
    sorted_pr = sorted( pr.items(), key = lambda x : x[ 1 ] )
    for i in range( 10 ):
        print( sorted_pr[ -1 - i ] )

def main():
    parser = argparse.ArgumentParser( description='pagerank', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '--datetime', type = str, default = '2021-06-03 21:19:03', help='datetime' )
    args = parser.parse_args()

    datetime = args.datetime
    timestamp = datetime2timestamp( datetime )
    # print( timestamp )
    
    process( timestamp )

if __name__ == "__main__":
    main()