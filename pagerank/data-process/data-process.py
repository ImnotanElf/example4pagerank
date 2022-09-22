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

if __name__ == "__main__":
    get_cpp_list()