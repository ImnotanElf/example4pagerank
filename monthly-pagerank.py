import time, datetime

def datetime2timestamp( datetime_str ): # datetime_str = '2021-06-03 21:19:03'
    # 转为时间数组
    timeArray = time.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    # print(timeArray)
    # timeArray可以调用tm_year等
    # print(timeArray.tm_year)
    #print(timeArray.tm_yday)
    # 转为时间戳
    timeStamp = int(time.mktime(timeArray))
    #print(timeStamp)
    return timeStamp

def timestamp2datetime( timestamp ):
    time_local = time.localtime(timestamp / 1000)
    print( time_local )
    # 转换成新的时间格式(精确到秒)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    print(dt) #2021-11-09 09:46:48
    d = datetime.datetime.fromtimestamp(timestamp / 1000)
    print( d )
    # 精确到毫秒
    str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(str1) #2021-11-09 09:46:48.000000

if __name__ == "__main__":
    timestamp2datetime( 1587983548 )