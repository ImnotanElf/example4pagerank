import glob

def merge():
    path = f"./labeled/*.txt"
    txt_list = glob.glob( path )
    print( u'Found %s txt file' % len( txt_list ) )
    print( u'Processing............' )
    with open( "./data/labeled.txt", 'w' ) as fw:
        for i in txt_list: 
            with open( i, 'r' ) as fr:
                lines = fr.readlines()
                fw.writelines( lines )
if __name__ == "__main__":
    merge()