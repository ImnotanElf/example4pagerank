import glob

def main():
    addr_set = set()
    with open( 'labeled', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].lower()
            # cate = info[ 1 ].lower()
            if addr not in addr_set:
                addr_set.add( addr )
    count = 0
    print( "Read success!" )
    with open( 'mapped-labeled.txt', 'w' ) as fw:
        for x in addr_set:
            fw.write( x + ',' + str( count ) + '\n' )
            count += 1
    print( "Write success!" )