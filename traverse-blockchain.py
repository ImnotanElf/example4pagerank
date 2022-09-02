# -*- coding:utf-8 -*-
from web3 import Web3
# only effective for rinkeby network
# from web3.middleware import geth_poa_middleware
from hexbytes import HexBytes
import argparse
import datetime, time
import pandas as pd
import csv
# import aiohttp
# import asyncio
import time
import multiprocessing
from multiprocessing import Process
import os, shutil


csv.field_size_limit(500 * 1024 * 1024)


class eth_api:
    def __init__(self, host, project_id):
        assert isinstance(host, str)
        assert isinstance(project_id, str)
        self.host = host
        self.wss = self.__websocket_provider(host, project_id)
        self.w3 = self.wss
        # only effective for rinkeby network
        # self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    def __websocket_provider(self, host, project_id):
        url = 'wss://%s.infura.io/ws/v3/%s' % (host, project_id)
        return Web3(Web3.WebsocketProvider(url))

    def get_block(self, symbol):
        return self.w3.eth.getBlock(symbol)

    def get_block_height(self):
        block = self.get_block('latest')
        assert 'number' in block
        return block['number']

    def get_transaction_receipt(self, txhash):
        return self.w3.eth.getTransactionReceipt(txhash)

    def get_transaction_by_hash(self, txhash):
        return self.w3.eth.getTransaction(txhash)

    def readonly_contract(self, contract_addr, data):
        # contract_addr = util.checksum_encode(contract_addr)
        return self.eth_call({'to': contract_addr, 'data': data})

    def eth_call(self, transaction):
        return self.w3.eth.call(transaction)

    def __check_filter_params(self, key, val, d):
        if val:
            d.update({key: val})
        return d

    def create_filter(self, block_s, block_e, address, topics):
        d = self.__check_filter_params('fromBlock', block_s, dict())
        d = self.__check_filter_params('toBlock', block_e, d)
        d = self.__check_filter_params('address', address, d)
        d = self.__check_filter_params('topics', topics, d)
        return self.wss.eth.filter(d)

    def get_filter_changes(self, filter_id):
        return self.wss.eth.getFilterChanges(filter_id)

    def get_filter_logs(self, filter_id):
        return self.wss.eth.getFilterLogs(filter_id)

    def get_code( self, hash_value ):
        return self.w3.eth.getCode( hash_value )

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

def main_process( start_number, end_number, args ): 
    # print( 1 )
    # url = 'wss://%s.infura.io/ws/v3/%s' % ( args.host, args.project_id )
    url = 'https://%s.infura.io/v3/%s' % ( args.host, args.project_id )
    # w3 = Web3( Web3.WebsocketProvider( url ) )
    w3 = Web3( Web3.HTTPProvider( url ) )

    res = []
    latest_block = w3.eth.getBlock( 'latest' )
    latest_number = latest_block[ 'number' ]
    # print( 2 )
    for i in range( start_number, end_number ):
        if i > latest_number:
            break
        block = w3.eth.getBlock( i, True )
        # print( 5 )
        try:
            assert 'transactions' in block
        except Exception as e:
            print( i, e )
            continue
        print( i )
        for tx in block[ 'transactions' ]:
            tx_hash = tx[ 'hash' ].hex()
            tx_from   = tx['from']
            tx_to     = tx['to']
            tx_value = tx['value']
            tx_blockNumber = tx[ 'blockNumber' ]
            tx_timestamp = block[ 'timestamp' ]
            tx_miner = block[ 'miner' ]
            res.append( [ str( tx_hash ), str( tx_from ), str( tx_to ), str( tx_value ), str( tx_blockNumber ), str( tx_timestamp ), str( tx_miner ) ] )   
    res_path = f"./outs/{ start_number }-{ end_number }.txt"
    with open( res_path, 'w', encoding='utf-8' ) as f:
        for tx in res:
            for tx_info in tx:
                f.write( tx_info+ ',' )
            f.write( '\n' )
    print( res_path )

def main():
    if not os.path.exists( "./outs" ):
        os.mkdir( "./outs" )
    else:
        shutil.rmtree( "./outs", ignore_errors=True)
        os.mkdir( "./outs" )

    parser = argparse.ArgumentParser(
        description='ETH API', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', type=str, default='rinkeby',
                        help='network host connection')
    parser.add_argument('--project_id', type=str,
                        default='', help='infura project id')
    parser.add_argument('--mp_number', type=int,
                        default=4, help='number of multiprocess')
    args = parser.parse_args()

    cpu_count = multiprocessing.cpu_count()
    print( "cpu_count: ", cpu_count )
    start = time.time()
    number_of_mp = args.mp_number
    print( "number of multiprocess: ", number_of_mp )
    raw_start_number = 291_2407
    raw_end_number   = 291_3000
    raw_amounts = raw_end_number - raw_start_number
    for j in range( 100 ):
        start_number = raw_start_number + j * int( raw_amounts / 100 )
        end_number = raw_start_number + ( j + 1 ) * int( raw_amounts / 100 )
        amounts = end_number - start_number
        _processes = []
        # print( 3 )
        for index in range( 0, number_of_mp ):
            _process = multiprocessing.Process( target=main_process, args=( start_number + int( index * amounts / number_of_mp ), start_number + ( index + 1 ) * int( amounts / number_of_mp ), args ) )
            _process.start()
            _processes.append(_process)
        for _process in _processes:
            _process.join()
    end = time.time()
    print( 'Cost time:', end - start )

if __name__ == '__main__':
    main()
