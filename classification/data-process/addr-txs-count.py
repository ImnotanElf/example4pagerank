import glob, argparse

def get_txs_count( train_rate ):
    assert( 0 < train_rate and train_rate < 1 )

    l_d = {}
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            cate = info[ 1 ]
            l_d[ addr ] = cate

    path = f"./outs/train-{ train_rate }/*.txt"
    txt_list = glob.glob( path )
    print(u'共发现%s个txt文件'% len( txt_list ) )
    print(u'正在处理............')
    labeled_dict = {}
    count = 0
    count_labeled_txs = 0
    for i in txt_list: 
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            count += len( lines )
            print( count )
            for line in lines:
                info = line.split( ',' )
                addr_from = info[ 0 ].lower()
                addr_to   = info[ 1 ].lower()
                if addr_from not in labeled_dict:
                    labeled_dict[ addr_from ] = 1
                else:
                    labeled_dict[ addr_from ] += 1
                if addr_to not in labeled_dict:
                    labeled_dict[ addr_to ] = 1
                else:
                    labeled_dict[ addr_to ] += 1
                count_labeled_txs += 1
    with open( f'addr-txs-count-{ train_rate }.txt', 'w' ) as fw:
        for k, v in labeled_dict.items():
            fw.write( k + ',' + ( str( v ) + ',' ).ljust( 6, ' ' ) + l_d[ k ] + '\n' )
    print( count, count_labeled_txs )

def main():
    parser = argparse.ArgumentParser( description='ETH API', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '--train_rate', type=float, default=0.5, help='train rate' )
    args = parser.parse_args()

    train_rate = args.train_rate
    
    get_txs_count( train_rate )

if __name__ == "__main__":
    main()