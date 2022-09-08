from email.policy import default
import time

def datetime2timestamp( datetime_str ): # datetime_str = '2021-06-03 21:19:03'
    timeArray = time.strptime( datetime_str, "%Y-%m-%d %H:%M:%S" )
    timeStamp = int( time.mktime( timeArray ) )
    return timeStamp

def timestamp2datetime( timestamp ): # unit: second
    time_local = time.localtime( timestamp )
    dt = time.strftime( "%Y-%m-%d %H:%M:%S", time_local )
    return dt

def get_stamps( every_months = 1 ):
    months = []  
    for x in range( 2023, 2016, -1 ):
        for y in range( 12, 0, -1 ):
            months.append( f"{ x }-{ str( y ).rjust( 2, '0' ) }-01 00:00:00" )
    stamps = [ datetime2timestamp( x ) for x in months ]
    nowstamp = time.time()
    month_index = 0
    for i in range( len( stamps ) ):
        if stamps[ i ] < nowstamp:
            month_index = i
            break
    res = []
    while ( month_index < len( stamps ) ):
        res.append( stamps[ month_index : month_index + every_months + 1 ] )
        month_index += every_months
    res[ 0 ][ 0 ] = nowstamp
    if ( len( res[ -1 ] ) == 1 ):
        res.pop()
    res = [ [ x[ 0 ], x[ -1 ] ] for x in res ]
    return res

if __name__ == "__main__":
    a = timestamp2datetime( 1587983548 )
    a = timestamp2datetime( time.time() )
    print( a )
    b = get_stamps( 3 )
    print( b )