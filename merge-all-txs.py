import glob

def merge():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    rate = 0
    all_line = []
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            rate += len( lines )
            print( "%.2f%%" % ( rate / 4152464.91 ) )
            for line in lines:
                all_line.append( line.strip() )
    print( "Read success!" )
    res = sorted( all_line, key = lambda x : int( x.split( ',' )[ -1 ] ), reverse = True )
    with open( 'all-txs.txt', 'w' ) as fw:
        for tx in res:
            fw.write( tx + '\n' )
    print( "Write success!" )
                

if __name__ == "__main__":
    merge()