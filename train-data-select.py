import math, glob, os, shutil, argparse

def select_train_data( train_rate ):
    cate_dict = {}
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            cate = info[ 1 ]
            if cate not in cate_dict:
                cate_dict[ cate ] = 1
            else:
                cate_dict[ cate ] += 1
        with open( f'train-{ train_rate }-labeled.txt', 'w' ) as fw:
            index = 0
            while ( index < len( lines ) ):
                info = lines[ index ].strip().split( ',' )
                addr = info[ 0 ].lower()
                cate = info[ 1 ]
                cate_length = cate_dict[ cate ]
                assert( 0 < train_rate and train_rate < 1 )
                cate_train_length = math.ceil( cate_length * train_rate )
                for i in range( cate_train_length ):
                    fw.write( lines[ index + i ] )
                index += cate_length

def select_train_txs( train_rate ):
    path = "/data/dev/outs/*.txt"
    txt_list = glob.glob( path )
    print(u'共发现%s个txt文件'% len( txt_list ) )
    print(u'正在处理............')
    labeled_dict = {}
    labeled_set = set()
    with open( 'train-labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            label = info[ 1 ]
            labeled_dict[ addr.lower() ] = label
            labeled_set.add( addr.lower() )
    count = 0
    count_labeled_txs = 0
    for i in txt_list: 
        with open( i, 'r' ) as fr:
            res_name = i.split( '/' )[ -1 ]
            res_path = f"./outs/train-{ train_rate }/" + res_name
            with open( res_path, 'w' ) as fw:
                lines = fr.readlines()
                count += len( lines )
                print( "{:.2f}%".format( count / 140912.07 ) )
                for line in lines:
                    info = line.strip().split( ',' )
                    if ( info[ 0 ].lower() in labeled_set and info[ 1 ].lower() in labeled_set ):
                        count_labeled_txs += 1
                        fw.write( line )
    print( count, count_labeled_txs )

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='ETH API', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '--train_rate', type=float, default=0.5, help='train rate' )
    args = parser.parse_args()

    train_rate = args.train_rate
    
    if not os.path.exists( f"./outs/train-{ train_rate }/" ):
        os.makedirs( f"./outs/train-{ train_rate }/" )
    else:
        shutil.rmtree( f"./outs/train-train-{ train_rate }/", ignore_errors=True)
        os.makedirs( f"./outs/train-train-{ train_rate }/" )
    select_train_data( train_rate )
    select_train_txs( train_rate )
