#coding=utf-8
import socket 
import sys
import time

# 设定每个数组包大小为128字节
# 统计 CSTEP 个数据包的结果， CSTEP越大计算越准确，
# CSTEP越小，计算不够准确，但是实验越快
BUFFER_SIZE = 128
TIME_OUT = 5

# python3 client.py %s %d %s %d %d %d %d" % (dst_host.IP(), dst_port, src_host.IP(), src_port, demand, rtime, time_step)

# python3 server.py %s %d %d %d %d" % (dst_host.IP(), dst_port, rtime, rtype, time_step)

if __name__== '__main__': 

    # 目标地址
    addr = sys.argv[1]
    # 目标端口
    port = int(sys.argv[2])
    # 运行时间
    rtime = int(sys.argv[3])
    # 流量类型
    rtype = int(sys.argv[4])

    time_step = int(sys.argv[5]) # not used

    # 绑定UDP地址
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((addr, port))
    
    ind = 0
    delay = 0
    throughput = 0
    time_stamp = time.time()
    start_time = time.time()

    # 通过CSTEP控制发送速率

    # 流量类型为0 每10个数据包统计一次
    if rtype == 0:
        CSTEP = 5 # small flow only need delay, small CSTEP can speed up the experiment

    # 流量类型为3 每50个数组包统计一次
    elif rtype == 3:
        CSTEP = 50

    # 其他流量类型 每30个数组包统计一次
    # 这里设定在没有流量区分的情况下，平均每30次丢包计算一次
    else:
        CSTEP = 30

    # if rtype == 0:
    #     CSTEP = 10  # small flow only need delay, small CSTEP can speed up the experiment
    #
    # else:
    #     CSTEP = 30

    ind_stamp = 0

    while True:

        try:

            # 设置缓冲区超时时间
            # 在网络通信中，如果对端没有响应，默认行为是等待数据。通过设置超时，可以避免程序长时间挂起。
            sock.settimeout(TIME_OUT)

            # 从缓冲区接收BUFFER_SIZE大小数据包
            data, addr = sock.recvfrom(BUFFER_SIZE)

        except socket.timeout:
            continue

        # 对发送的数据包进行解码，
        # 第一个数据为数据包编号、第二个为发送数据包的时间戳
        infos = str(data.decode()).split(';')[:-1]

        # 累加延迟
        delay += int(time.time() * 1000) - int(infos[1])

        # 累加缓冲区, BUFFER_SIZE是字节数大小， *8 表示bit数大小
        # 这里除以发送数据包间隔时间得到吞吐量大小 kbps
        throughput += BUFFER_SIZE * 8

        if ind % CSTEP == 0 and ind != 0:
            # since no packet disorder in simulate environment
            # in fact we only need the first several records too much record(about 1000 records) will crash popen buffer and make the server killed
            if ind / CSTEP <= 10:
                # 计算CSTEP个数据包的 平均延迟、吞吐和丢包
                # 关于吞吐量的统计 使用累计的接收缓冲区大小的数据包 除以时间
                # 这里关于丢包的数据统计不够准确 计算的是与上次ind_stamp相差的数据包个数
                print("delay: %f ms throughput: %f Kbps loss_rate: %f" % (delay / CSTEP, throughput / 1e3 / (time.time() - time_stamp), (int(infos[0]) - ind_stamp - CSTEP) / (int(infos[0]) - ind_stamp)), flush=True) # to flush the content to popen and pmonitor
            delay = 0
            throughput = 0
            time_stamp = time.time()
            ind_stamp = int(infos[0])

        ind += 1

    sock.close()

