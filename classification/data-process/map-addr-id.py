import glob

def main():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print( u'共发现%s个txt文件' % len( txt_list ) )
    print( u'正在处理............' )
    addr_set = set()
    rate = 0
    for i in txt_list: #循环读取同文件夹下的txt文件
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            rate += len( lines )
            print( "{:0.2f}%".format( rate / 11497057.07 ) )
            for line in lines:
                info = line.strip().split( ',' )
                addr_from = info[ 0 ].lower()
                addr_to   = info[ 1 ].lower()
                if addr_from not in addr_set:
                    addr_set.add( addr_from )
                if addr_to not in addr_set:
                    addr_set.add( addr_to )
    count = 0
    print( "Read success!" )
    with open( 'mapped.txt', 'w' ) as fw:
        for x in addr_set:
            fw.write( x + ',' + str( count ) + '\n' )
            count += 1
    print( "Write success!", rate )

if __name__ == "__main__":
    main()