#coding=utf-8
import socket
import sys
import time

BUFFER_SIZE = 128

def padding_bytes(x, target_len):
    clen = len(x)
    x += bytes(target_len - clen)
    return x

# python3 client_s.py %s %d %s %d %d %d %d" % (dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step)

# python3 server_s.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step)

if __name__ == '__main__':

    # 服务器地址
    server_addr = sys.argv[1]
    # 服务器端口
    server_port = int(sys.argv[2])
    # 客户端地址
    client_addr = sys.argv[3]
    # 客户端端口
    client_port = int(sys.argv[4])
    # 流量大小 Kbps
    demand = int(sys.argv[5]) # Kbps
    # 运行时间 s
    rtime = int(sys.argv[6]) # seconds

    time_step = int(sys.argv[7]) #for testing 

    # TCP套接字 socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # UDP套接字 socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((client_addr, client_port))

    ind = 0
    start_time = time.time()
    time_stamp = int(time.time() * 1000)

    while True:

        # 获取当前时间戳
        temp_stamp = time.time()

        # 数据包填充 ind 编号和时间戳
        msg = "%d;%d;" % (ind, int(temp_stamp * 1000))

        # 将数据包填充至128字节，也就是发送一个数据包是128字节
        msg = padding_bytes(msg.encode(), BUFFER_SIZE)

        # 向服务器端口发送128字节的数据包
        sock.sendto(msg, (server_addr, server_port))
        
        ind += 1
        # 计算当前比特数
        curr_bit = ind * BUFFER_SIZE * 8
        # 获取当前时间戳
        temp_stamp = time.time()

        # 控制当前数据包发送速率
        # self._request_demands = [[100], [1500], [1500], [500]]
        # 如果不设定流量类型，直接随机从中选择
        # 4种类型的流量 100kbps、1500kbps、1500kbps、500kbps

        # 当前传输bit数超过带宽需求
        # 带宽数据是 kbps大小，
        if curr_bit > (temp_stamp - start_time) * demand * 1000:
            # 等待时间重新发送，等待一个数据包大小的时间
            time.sleep(BUFFER_SIZE / (demand * 125))

    sock.close()

