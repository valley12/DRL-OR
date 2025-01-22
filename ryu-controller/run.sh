# 设置目标端口
TARGET_PORT=3999

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

#ryu-manager controller.py --ofp-tcp-listen-port 5001 --config-file ./config_files/GEA.config

# Abi Topo
ryu-manager controller.py --ofp-tcp-listen-port 5001 --config-file ./config_files/Abi.config