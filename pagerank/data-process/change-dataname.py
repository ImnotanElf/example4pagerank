import glob


def get_txt_list():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    # sorted( txt_list, key = lambda x : int( x.split( '/' )[ -1 ].split( '-' )[ 0 ] ) )
    return txt_list

def process():
    txt_list = get_txt_list()
    print( txt_list )
    count = 0
    for i in txt_list:
        with open( i, 'r' ) as fr:
            count += 1
            print( count )
            lines = fr.readlines()
            info_last  = lines[ -1 ].strip().split( ',' )
            info_first = lines[  0 ].strip().split( ',' )
            with open( f'./data/{ info_first[ 4 ] }-{ info_last[ 4 ] }.txt', 'w' ) as fw:
                fw.writelines( lines )

if __name__ == "__main__":
    process()