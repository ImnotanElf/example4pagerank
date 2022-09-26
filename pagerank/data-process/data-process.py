
import time, glob

def datetime2timestamp( datetime_str ): # datetime_str = '2021-06-03 21:19:03'
    timeArray = time.strptime( datetime_str, "%Y-%m-%d %H:%M:%S" )
    timeStamp = int( time.mktime( timeArray ) )
    return timeStamp

def timestamp2datetime( timestamp ): # unit: second
    time_local = time.localtime( timestamp )
    dt = time.strftime( "%Y-%m-%d %H:%M:%S", time_local )
    return dt

def get_cpp_list():
    path = "/data/dev/pagerank/outs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    edges_list = []
    dict_list  = []
    for i in txt_list:
        if i.split( '-' )[ -1 ] == "cpp.txt" and i.split( '-' )[ -2 ] == "edges" :
            edges_list.append( i )
        if i.split( '-' )[ -1 ] == "cpp.txt" and i.split( '-' )[ -2 ] == "dict" :
            dict_list.append( i )
    with open( './cpp/data/edges-list.txt', 'w' ) as fw:
        for edges in edges_list:
            fw.write( edges + '\n' )
    with open( './cpp/data/dict-list.txt', 'w' ) as fw:
        for d in dict_list:
            fw.write( d + '\n' )

def get_txt_list():
    path = "/data/dev/pagerank/data/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    txt_list = sorted( txt_list, key = lambda x : x.split( '/' )[ -1 ].split( '-' )[ 0 ] )
    with open( './cpp/data/txt-list.txt', 'w' ) as fw:
        for txt in txt_list:
            fw.write( txt + '\n' )
    return txt_list

def get_stamps():
    start_datetime = '2017-01-01 00:00:00'
    start_stamp = datetime2timestamp( start_datetime )
    stamps_span_42days = 42 * 24 * 3600
    stamps = []
    for i in range( 51 ):
        stamps.append( start_stamp + i * stamps_span_42days )
    with open( './cpp/data/stamps-datetimes.txt', 'w' ) as fw:
        for s in stamps:
            fw.write( str( s ) + ',' + str( timestamp2datetime( s ) ) + '\n' )
    return stamps

def get_final_list():
    path = "/data/dev/pagerank/outs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    dict_list  = []
    for i in txt_list:
        if i.split( '-' )[ -1 ] == "cpp.txt" and i.split( '-' )[ -2 ] == "dict" :
            dict_list.append( i )

    path = "/data/dev/pagerank/outs/*.out"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个out文件' % len( txt_list ) )
    print( u'正在处理............' )
    outs_list = []
    for i in txt_list:
        if i.split( '-' )[ -1 ] == "cpp.txt.out" and i.split( '-' )[ -2 ] == "edges" :
            outs_list.append( i )
    
    print( len( outs_list ) )
    print( len( dict_list ) )
    test_count = 0
    for i in outs_list:
        print( test_count )
        test_count += 1
        for j in dict_list:
            if i.split( "---" )[ 0 ] == j.split( "---" )[ 0 ]:
                id_addr_d = {}
                with open( j, 'r' ) as frj:
                    j_lines = frj.readlines()
                    for index in range( len( j_lines ) ):
                        info = j_lines[ index ].split( ',' )
                        addr = info[ 0 ]
                        id_addr_d[ index ] = addr
                with open( i, 'r' ) as fri:
                    res_d = {}
                    i_lines = fri.readlines()
                    try:
                        assert( len( i_lines ) == len( id_addr_d ) )
                    except:
                        print( i )
                        print( j )
                        print( len( i_lines ) )
                        print( len( id_addr_d ) )
                    for index in range( len( i_lines ) ):
                        if index in id_addr_d:
                            res_d[ id_addr_d[ index ] ] = float( i_lines[ index ].strip() )
                    s_pr = sorted( res_d.items(), key = lambda x : x[ 1 ], reverse = True )
                    with open( j + ".finalout", 'w' ) as fw:
                        for pr in s_pr:
                            fw.write( str( pr[ 0 ] ) + ',' + str( pr[ 1 ] ) + '\n' )   

def get_top_n( top_n ):
    path = "/data/dev/pagerank/outs/finalout/*.finalout"
    csv_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个csv文件' % len( csv_list ) )
    print( u'正在处理............' )
    for i in csv_list:
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            with open( i + f".top{ top_n }", 'w' ) as fw:
                for i in range( top_n ):
                    fw.write( lines[ i ] )

def merge_csv():
    path = "/data/dev/pagerank/outs/finalout/top_n/*.top100"
    csv_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个csv文件' % len( csv_list ) )
    print( u'正在处理............' )
    with open( "merge_top100.csv", 'w' ) as fw:
        for i in csv_list:
            with open( i, 'r' ) as fr:
                date_t = i.split( '/' )[ -1 ].split( "-dict" )[ 0 ]
                lines = fr.readlines()    
                for line in lines:
                    fw.write( line.strip() + ',' + date_t + '\n' )

def generate_edges():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个out文件' % len( txt_list ) )
    print( u'正在处理............' )
    ad2id = {}
    fw_ad = open( 'all-ad2id-final.csv', 'w' )
    ad_count = 0
    ed_count = 0
    te_count = 0
    txt_count = 0
    with open( 'all-edges.csv', 'w' ) as fw_ed:
        for i in txt_list:
            assert( ad_count == len( ad2id ) )
            with open( i, 'r' ) as fr:
                print( txt_count )
                txt_count += 1
                lines = fr.readlines()
                for j in range( len( lines ) ):
                    assert( ad_count == len( ad2id ) )
                    assert( "none" not in ad2id )
                    ed_count += 1
                    info = lines[ j ].strip().split( ',' )
                    tx_from  = info[ 0 ].lower()
                    tx_to    = info[ 1 ].lower()
                    tx_value = int( info[ 2 ] )
                    tx_blockheight = int( info[ 3 ] )
                    if tx_from == "none" and tx_to == "none":
                        print( "wtf" )
                        ee = 1 / 0
                    if tx_from == 'none' and tx_to != 'none':
                        if tx_to not in ad2id:
                            ad2id[ tx_to ] = ad_count
                            fw_ad.write( str( ad_count ) + '\n' )
                            ad_count += 1
                        fw_ed.write( str( ad2id[ tx_to ]  ) + ',' + str( ad2id[ tx_to ] ) + '\n' )
                        te_count += 1
                        continue
                    if tx_from != 'none' and tx_to == 'none':
                        if tx_from not in ad2id:
                            ad2id[ tx_from ] = ad_count
                            fw_ad.write( str( ad_count ) + '\n' )
                            ad_count += 1
                        fw_ed.write( str( ad2id[ tx_from ]  ) + ',' + str( ad2id[ tx_from ] ) + '\n' )
                        te_count += 1
                        continue
                    if tx_to not in ad2id:
                        ad2id[ tx_to ] = ad_count
                        fw_ad.write( str( ad_count ) + '\n' )
                        ad_count += 1
                    if tx_from not in ad2id:
                        ad2id[ tx_from ] = ad_count
                        fw_ad.write( str( ad_count ) + '\n' )
                        ad_count += 1
                    fw_ed.write( str( ad2id[ tx_from ]  ) + ',' + str( ad2id[ tx_to ] ) + '\n' )
                    te_count += 1
    print( "Read done!" )
    print( "number of addrs: ", len( ad2id ) )
    print( "number of addrs: ", ad_count )
    print( "number of edges: ", ed_count )
    print( "number of tests: ", te_count )
    fw_ad.close()
    with open( 'all-ad2id.csv', 'w' ) as fw:
        for k, v in ad2id.items():
            fw.write( str( k ) + ',' + str( v ) + '\n' )
    print( "Write all-ad2id.csv done!" )


if __name__ == "__main__":
    # generate_edges()
    get_final_list()