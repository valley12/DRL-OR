sudo mn -c

sudo lsof -i:5000 | awk '{print $2}' | awk 'NR==2{print}' | xargs kill -9

#!/bin/bash

# 设置目标端口
TARGET_PORT=5000

# 检查端口5000是否被占用
is_port_in_use() {
    netstat -tuln | grep ":$1" > /dev/null
}

# 如果端口被占用，结束占用该端口的进程
kill_process_using_port() {
    local pid=$(lsof -t -i :$1)
    if [ ! -z "$pid" ]; then
        echo "Port $1 is in use by process ID $pid. Killing the process."
        kill -9 $pid
    else
        echo "Port $1 is not in use."
    fi
}

# 执行操作
echo "Checking if port $TARGET_PORT is in use..."
if is_port_in_use $TARGET_PORT; then
    kill_process_using_port $TARGET_PORT
else
    echo "Port $TARGET_PORT is not in use."
fi


sudo python3 testbed.py Abi
#sudo python3  testbed.py GEA
