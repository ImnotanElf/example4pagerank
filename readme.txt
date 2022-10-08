此部分有两个部分的工作，classification 和 pagerank
与classification相关的工作已全部移交给张金波（ 绰号大佬 ），如有相关疑问可向其咨询。

下面叙述pagerank相关的工作：
0、如有数据更新，下载完数据之后，存放到指定位置
   cd /data/dev/pagerank/
   bash build.sh
   生成文件：
   merge_top100.csv
   此文件路径为 10.10.10.10:/data/dev/pagerank/outs/finalout/top_n/merge_top100.csv
   此文件为典枢算法上传的 源数据文件
   具体程序设计，详情如下：

1、数据下载
数据下载请联系 王宸敏， 因为国内的服务器没办法连接到 mainnet

2、数据存储
数据存储于 10.10.10.10:/data/ethereum-data/txs

3、数据处理
所有的处理程序均在10.10.10.10:/data/dev/pagerank/路径下面
a. change-dataname.py
   如有新的数据添加，只需执行
   python3 change-dataname.py
   此程序用于，将下载下来的包含交易的txt文件更改名字，将原文件名称里的 区块链节点高度（ block_height ）替换为等效的时间戳（ timestamp ），以方便后续根据时间戳查找
b. monthly-pagerank-for-cpp.py
   如有的数据添加，只需执行
   python3 monthly-pagerank-for-cpp.py
   此程序用于，将下载下来的交易数据，根据时间节点（ 目前的程序为，自2017年07月01日开始，每隔42个自然日为一个周期，计算该时间段内各节点的pagerank数值 ）
   每42天存为一个文件，输出文件存储在 10.10.10.10:/data/dev/pagerank/outs 路径当中的 *-edges-cpp.txt
   文件内容为：
   每行两个数值，代表一个交易的源节点和目标节点，每个数值为一个账户节点的id，id与账户的对应关系存放在 *-dict-cpp.txt
c. 进入 10.10.10.10:/data/dev/pagerank/cpp/
   执行 make
   生成二进制可执行文件 pagerank
   执行 ./pagerank -n -d "," -c 0.000001 -m 100 ss
   此程序用于，调用pagerank算法，生成每个节点的pagerank数值，存放于 10.10.10.10:/data/dev/pagerank/outs 路径当中的 *-edges-cpp.txt.out
   文件内容为，每行为一个节点的pagerank数值，第一行为节点1对应的pagerank数值，第二行为节点2对应的pagerank数值，依此类推
   id与账户的对应关系存放在 *-dict-cpp.txt
d. 回到 10.10.10.10:/data/dev/pagerank/
   执行
   python3 data-process.py
   此程序用于，在生成的 *-edges-cpp.txt.out 文件中添加账户节点地址，如，第一行添加id为1对应的账户地址
   完了按照pagerank数值倒序排序
   生成 *-edges-cpp.txt.out.finalout
e. 进入 10.10.10.10:/data/dev/pagerank/outs

    if [ ! -d "/data/dev/pagerank/outs/finalout/" ];then
        mkdir -p /data/dev/pagerank/outs/finalout/"
    else
        echo "finalout文件夹已经存在"
    fi
    将文件夹下所有的.finalout文件移入到./finalout/
f. 进入 10.10.10.10:/data/dev/pagerank/outs/finalout/
    执行
    python3 data-process.py
    获得每个42天的pagerank值top100账户节点

    if [ ! -d "/data/dev/pagerank/outs/finalout/top_n/" ];then
        mkdir -p /data/dev/pagerank/outs/finalout/top_n/"
    else
        echo "top_n文件夹已经存在"
    fi
    将文件夹下所有的.top100文件移入到./top_n/

g. 进入 10.10.10.10:/data/dev/pagerank/outs/finalout/top_n/
    执行
    python3 data-process.py
    将所有csv文件合并
    生成merge_top100.csv
    此文件为典枢算法上传的 源数据文件

4、典枢算法编写
   此处算法编写比较简单，基本就是查询返回。