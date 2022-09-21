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

def get_stamps_deprecated( every_months = 1 ):
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

def get_stamps():
    start_datetime = '2017-01-01 00:00:00'
    start_stamp = datetime2timestamp( start_datetime )
    stamps_span_42days = 42 * 24 * 3600
    stamps = []
    for i in range( 51 ):
        stamps.append( start_stamp + i * stamps_span_42days )
    return stamps

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

def get_txt_list():
    path = "./data/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    txt_list = sorted( txt_list, key = lambda x : x.split( '/' )[ -1 ].split( '-' )[ 0 ] )
    return txt_list

def is_ordered( nums ):
    if len( nums ) < 2:
        return True
    for i in range( len( nums ) - 1 ):
        if ( nums[ i + 1 ] < nums[ i ] ):
            return False
    return True

def binary_search( nums, target_stamp, left, right, is_smaller ):
    while ( left < right ):
        mid = left + ( right - left ) // 2
        mid_stamp = nums[ mid ]
        if is_smaller:
            if target_stamp < mid_stamp:
                right = mid
            else:
                left = mid + 1
        else:
            if mid_stamp < target_stamp:
                left = mid + 1
            else:
                right = mid
    return max( 0, left - 1 ) if is_smaller else min( right, len( nums ) - 1 )

def process():
    txt_list = get_txt_list()
    nums = [ int( x.strip().split( '/' )[ -1 ].split( '-' )[ 0 ] ) for x in txt_list ]
    assert( is_ordered( nums ) )
    
    timestamps = get_stamps()

    for i in range( 8, len( timestamps ) - 1 ):
        G1 = nx.Graph()
        count = 0
        
        start_stamp = timestamps[ i ]
        end_stamp = timestamps[ i + 1 ]
        start_datetime = timestamp2datetime( start_stamp )
        end_datetime = timestamp2datetime( end_stamp )

        print( "start-->end: ", start_datetime, "-->", end_datetime )

        index_left  = binary_search( nums, start_stamp, 0, len( txt_list ), is_smaller = True )
        print( "left:  ", index_left )
        index_right = binary_search( nums, end_stamp,   0, len( txt_list ), is_smaller = False )
        print( "right: ", index_right )
        for j in range( index_left, index_right + 1 ):
            with open( txt_list[ j ], 'r' ) as fr:
                lines = fr.readlines()
                for line in lines:
                    info = line.strip().split( ',' )
                    tx_from  = info[ 0 ].lower()
                    tx_to    = info[ 1 ].lower()
                    tx_value = int( info[ 2 ] )
                    tx_blockheight = int( info[ 3 ] ) 
                    tx_timestamp = int( info[ 4 ] )
                    if tx_timestamp < start_stamp or end_stamp < tx_timestamp:
                        continue
                    if tx_from == 'none' and tx_to != 'none':
                        count += 1
                        G1.add_edge( tx_to, tx_to, weight = 1 )
                        continue
                    if tx_from != 'none' and tx_to == 'none':
                        count += 1
                        G1.add_edge( tx_from, tx_from, weight = 1 )
                        continue
                    G1.add_edge( tx_from, tx_to, weight = 1 )
        print( "Number of nodes:", len( G1.nodes() ) )
        print( "count: ", count )

        pr = nx.pagerank(G1, alpha=0.85, personalization=None,
                    max_iter=100000, tol=1.0e-6, nstart=None, weight='weight',
                    dangling=None)
        sorted_pr = sorted( pr.items(), key = lambda x : x[ 1 ] )
        try:
            for i in range( 10 ):
                print( sorted_pr[-1 - i] )
        except Exception as e:
            print( e )
        with open( f"./outs/{ start_datetime.replace( ' ', '-' ) }---{ end_datetime.replace( ' ', '-' ) }.txt", 'w' ) as fw:
            for i in range( len( sorted_pr ) ):
                fw.write( str( sorted_pr[ -1 - i ] ) + '\n' )
        with open( f"./outs/{ start_stamp }-{ end_stamp }.txt", 'w' ) as fw:
            for i in range( len( sorted_pr ) ):
                fw.write( str( sorted_pr[ -1 - i ] ) + '\n' )
        G1.clear()
        print( "Number of nodes:", len( G1.nodes() ) )


def main():
    parser = argparse.ArgumentParser( description='pagerank', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '--datetime', type = str, default = '2021-06-03 21:19:03', help='datetime' )
    args = parser.parse_args()

    datetime = args.datetime
    timestamp = datetime2timestamp( datetime )
    # print( timestamp )
    
    process()

if __name__ == "__main__":
    main()
    # last_stamp = 1660716295
    # last_datetime = timestamp2datetime( last_stamp )
    # print( last_datetime ) # 2022-08-17 14:04:55