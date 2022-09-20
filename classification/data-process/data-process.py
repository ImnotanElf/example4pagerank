import glob

def labeled_info():
    labeled_dict = {}
    addr_id_dict = {}
    id_addr_dict = {}
    cate_id_dict = {}
    id_cate_dict = {}
    aici_dict = {} # AddrId_CateId NICI
    with open( 'labeled.txt', 'r' ) as fr:
        lines = fr.readlines()
        addr_id = 0
        cate_id = 0
        for line in lines:
            info = line.strip().split( ',' )
            addr = info[ 0 ].strip().lower()
            cate = info[ 1 ].strip()

            if addr not in addr_id_dict:
                addr_id_dict[ addr ]    = addr_id
                id_addr_dict[ addr_id ] = addr
                addr_id += 1
            if cate not in cate_id_dict:
                cate_id_dict[ cate ]    = cate_id
                id_cate_dict[ cate_id ] = cate
                cate_id += 1

            labeled_dict[ addr ] = cate
            aici_dict[ addr_id_dict[ addr ] ] = cate_id_dict[ cate ]
    return labeled_dict, addr_id_dict, cate_id_dict, aici_dict, id_addr_dict, id_cate_dict

def read_data():
    labeled_dict, addr_id_dict, cate_id_dict, aici_dict, id_addr_dict, id_cate_dict = labeled_info()

    edges = []
    addr_contents = {}
    id_contents = {}

    path = f"./outs/*.txt"
    txt_list = glob.glob( path )
    print( u'Found %s txt file' % len( txt_list ) )
    print( u'Processing............' )
    count = 0
    for i in txt_list: 
        with open( i, 'r' ) as fr:
            lines = fr.readlines()
            count += len( lines )
            print( "{:.2f}%".format( count / 140912.07 ) )
            for line in lines:
                info = line.split( ',' )
                tx_from  = info[ 0 ].strip().lower()
                tx_to    = info[ 1 ].strip().lower()
                tx_value = info[ 2 ].strip()
                tx_blockheight = info[ 3 ].strip()
                tx_timestamp   = info[ 4 ].strip()

                tmp_edge = {}
                tmp_edge[ "from" ]  = addr_id_dict[ tx_from ]
                tmp_edge[ "to" ]    = addr_id_dict[ tx_to ]
                tmp_edge[ "value" ] = tx_value
                tmp_edge[ "blockheight" ] = tx_blockheight
                tmp_edge[ "timestamp" ]   = tx_timestamp
                edges.append( tmp_edge )

                tmp_addr_features = {}
                if tx_from not in tmp_addr_features:
                    tmp_addr_features[ "addr" ] = tx_from
                    tmp_addr_features[ "id" ]   = addr_id_dict[ tx_from ]
                    tmp_addr_features[ "outdegree" ] = 1
                    tmp_addr_features[ "sumout" ]    = int( tx_value ) # sum values of out directed edge
                    tmp_addr_features[ "indegree" ]  = 0
                    tmp_addr_features[ "sumin" ]     = 0               # sum values of in directed edge
                    tmp_addr_features[ "addrtype" ]  = 1               # 1 for EOA ( Externally Owned Address ), 0 for CA ( Contract Address )
                    tmp_addr_features[ "cate" ]      = labeled_dict[ tx_from ]
                else:
                    tmp_addr_features[ "outdegree" ] += 1
                    tmp_addr_features[ "sumout" ]    += int( tx_value )
                addr_contents[ tx_from ] = tmp_addr_features
                id_contents[ addr_id_dict[ tx_from ] ] = tmp_addr_features
                
                tmp_addr_features = {}
                if tx_to not in tmp_addr_features:
                    tmp_addr_features[ "addr" ] = tx_to
                    tmp_addr_features[ "id" ]   = addr_id_dict[ tx_to ]
                    tmp_addr_features[ "outdegree" ] = 0
                    tmp_addr_features[ "sumout" ]    = 0
                    tmp_addr_features[ "indegree" ]  = 1
                    tmp_addr_features[ "sumin" ]     = int( tx_value ) 
                    tmp_addr_features[ "addrtype" ]  = 1
                    tmp_addr_features[ "cate" ]      = labeled_dict[ tx_to ]
                else:
                    tmp_addr_features[ "indegree" ] += 1
                    tmp_addr_features[ "sumin" ]    += int( tx_value )
                addr_contents[ tx_to ] = tmp_addr_features
                id_contents[ addr_id_dict[ tx_to ] ] = tmp_addr_features
    return edges, addr_contents, id_contents

def generate_data():
    edges, addr_contents, id_contents = read_data()
    sorted( edges, key = lambda x : ( int( x[ "from" ] ), int( x[ "to" ] ), int( x[ "blockheight" ] ) ) )
    with open( "./data/ethereum.edges", "w" ) as fw:
        for edge in edges:
            print( edge[ "from" ], edge[ "to" ] )
            fw.write( str( edge[ "from" ] ) + " " + str( edge[ "to" ] )  + " " + str( edge[ "value" ] ) + " " + str( edge[ "blockheight" ] ) + "\n" )
    with open( "./data/ethereum.contents", "w" ) as fw:
        for k, content in id_contents.items():
            print( content[ "id" ] )
            fw.write( str( content[ "id" ] ) + " " + 
                      str( content[ "outdegree" ] ) + " " + 
                      str( content[ "sumout" ] ) + " " + 
                      str( content[ "indegree" ] ) + " " + 
                      str( content[ "sumin" ] ) + " " + 
                      str( content[ "addrtype" ] ) + " " + 
                      str( content[ "cate" ] ) +  "\n" )

def main():
    generate_data()

if __name__ == "__main__":
    main()