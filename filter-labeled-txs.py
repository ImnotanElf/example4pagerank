import glob, os, shutil

def main():
    path = "/data/ethereum-data/txs/*.txt"
    txt_list = glob.glob( path )
    print(u'共发现%s个txt文件'% len( txt_list ) )
    print(u'正在处理............')
    labeled_dict = {}
    labeled_set = set()
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ]
            label = info[ 1 ]
            labeled_dict[ addr.lower() ] = label
            labeled_set.add( addr.lower() )
    count = 0
    count_labeled_txs = 0
    for i in txt_list: 
        with open( i, 'r' ) as fr:
            res_name = i.split( '/' )[ -1 ]
            res_path = "./outs/" + res_name
            with open( res_path, 'w' ) as fw:
                lines = fr.readlines()
                count += len( lines )
                print( "%0.2f %" % count / 4152464.91 )
                for line in lines:
                    info = line.split( ',' )
                    if ( info[ 0 ].lower() in labeled_set and info[ 1 ].lower() in labeled_set ):
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