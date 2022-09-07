import glob, os, shutil

def main():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path ) #查看同文件夹下的txt文件数
    print(u'共发现%s个CSV文件'% len( txt_list ) )
    print(u'正在处理............')
    labeled_dict = {}
    labeled_set = set()
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ]
            label = info[ 1 ]
            labeled_dict[ addr ] = label
            labeled_set.add( addr )
    res = []
    count = 0
    count_labeled_txs = 0
    for i in txt_list: #循环读取同文件夹下的csv文件
        with open( i, 'r' ) as fr:
            res_name = i.split( '/' )[ -1 ]
            res_path = "./outs/" + res_name
            with open( res_path, 'w' ) as fw:
                lines = fr.readlines()
                count += len( lines )
                for line in lines:
                    info = line.split( ',' )
                    if ( info[ 0 ] in labeled_set or info[ 1 ] in labeled_set ):
                        count_labeled_txs += 1
                        fw.write( line )
    print( count, count_labeled_txs )
    
if __name__ == "__main__":
    if not os.path.exists( "./outs" ):
        os.mkdir( "./outs" )
    else:
        shutil.rmtree( "./outs", ignore_errors=True)
        os.mkdir( "./outs" )
    main()