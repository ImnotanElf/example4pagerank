def add():
    d = {}
    with open( './tmp', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            addr = line.split( ',' )[ 0 ].lower()
            adtype = line.split( ',' )[ 1 ].strip()
            d[ addr ] = adtype
    with open( './data/labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        for line in lines:
            addr = line.split( ',' )[ 0 ].lower()
            adtype = line.split( ',' )[ 2 ].strip()
            d[ addr ] = adtype
    with open( './data/addrs.txt', 'w' ) as fw:
        with open( 'labeled.txt', 'r' ) as fr:
            lines = fr.readlines()
            for line in lines:
                addr = line.split( ',' )[ 0 ].lower()
                fw.write( line.strip() + ',' + d[ addr ] + '\n' )
    

if __name__ == "__main__":
    add()