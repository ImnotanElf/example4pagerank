def minus():
    x = set()
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            addr = line.split( ',' )[ 0 ].lower()
            x.add( addr )
    y = set()
    with open( './data/labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            addr = line.split( ',' )[ 0 ].lower()
            y.add( addr )
    z = x - y
    with open( 'to-be-added.txt', 'w' ) as fw:
        for i in z:
            fw.write( i + '\n' )
    print( len( x ), len( y ), len( z ) )

if __name__ == "__main__":
    minus()