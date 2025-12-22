#-----------------------------------------------
# 题目（请先完整读一遍）#
#写一个 Shell 脚本，完成下面功能：#
#功能要求#
#接收 一个参数：日志文件路径#
#如果文件不存在，打印错误并退出（exit code ≠ 0）#
#从日志中：#
#找出 包含 ERROR 的行#
#统计 ERROR 行数#
#如果 ERROR 数量 > 0：#
#打印：ERROR found: <count>#
#退出码 = 1#
#如果 ERROR 数量 = 0：#
#打印：No error found#
#退出码 = 0

#!/usr/bin/env bash
set -euo pipefail
## 给脚本上“安全带”
#-e：命令失败就退出
#-u：用到没定义的变量就报错
#pipefail：管道里任何一步失败都算失败
#log_file="$1"

# 参数检查
if [ -z "$log_file" ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

# 文件检查
if [ ! -f "$log_file" ]; then
    echo "File not found: $log_file"
    exit 1
fi

# 统计 ERROR 行数
error_count=$(grep "ERROR" "$log_file" | wc -l)

# 判断结果
if [ "$error_count" -gt 0 ]; then
    echo "ERROR found: $error_count"
    exit 1
else
    echo "No error found"
    exit 0
fi

#------------------------------------------------------
#写一个 shell 脚本，检查 指定网络接口是否存在且为 UP 状态
#要求：
#脚本接收 一个参数：接口名（如 eth0）
#如果参数为空 → 打印 usage，退出 exit 1
#如果接口不存在 → 打印错误，退出 exit 1
#如果接口存在但不是 UP → 打印状态，退出 exit 1
#如果接口存在且为 UP → 打印 OK，退出 exit 0
#!/usr/bin/env bash
set -euo pipefail
iface="$1"
# 1. 参数检查
if [ -z "$iface" ]; then
    echo "Usage: $0 <interface>"
    exit 1
fi
# 2. 接口是否存在
if ! ip link show "$iface" &>/dev/null; then
    echo "Interface not found: $iface"
    exit 1
fi
# 3. 接口是否为 UP
if ip link show "$iface" | grep -q "UP"; then
    echo "Interface $iface is UP"
    exit 0
else
    echo "Interface $iface is DOWN"
    exit 1
fi

#----------------------------------------------------------
已知 ip -s link show eth0 的输出如下格式（示例）：
RX: bytes  packets  errors  dropped
    12345  1000     0       12
TX: bytes  packets  errors  dropped
    67890  900      0       3

要求：
用 Shell 脚本统计 RX dropped + TX dropped
如果总丢包 > 0
打印：Packet loss detected: <count>
exit 1
如果没有丢包
打印：No packet loss
exit 0

#!/usr/bin/env bash
set -euo pipefail
iface="$1"
if [ -z "$iface" ]; then
    echo "Usage: $0 <interface>"
    exit 1
fi
output=$(ip -s link show "$iface")
rx_drop=$(echo "$output" | awk '/RX:/ {getline; print $4}')
tx_drop=$(echo "$output" | awk '/TX:/ {getline; print $4}')
#/RX:/👉 找到包含 RX: 的那一行
#getline👉 再读下一行
#print $4👉 打印第 4 列（dropped）
total_drop=$((rx_drop + tx_drop))
if [ "$total_drop" -gt 0 ]; then
    echo "Packet loss detected: $total_drop"
    exit 1
else
    echo "No packet loss"
    exit 0
fi

#-----------------------------------------
给定多个网络接口名，批量检查：
是否存在
是否 UP
只要 有一个接口失败
👉 脚本整体失败（exit 1）

#!/usr/bin/env bash
set -euo pipefail
check_iface() {
    local iface="$1"
    if ! ip link show "$iface" &>/dev/null; then
        echo "Interface not found: $iface"
        return 1
    fi
    if ip link show "$iface" | grep -q "UP"; then
        echo "$iface is UP"
        return 0
    else
        echo "$iface is DOWN"
        return 1
    fi
}
ifaces=("eth0" "eth1" "eth2")
for iface in "${ifaces[@]}"; do
    check_iface "$iface" || exit 1
done
echo "All interfaces are OK"
exit 0

#--------------------------------------------------------------------
写一个 sanity test 脚本：
检查接口 eth0 是否 UP
检查是否有丢包
任一失败 → 整体失败
全部通过 → 成功

#!/usr/bin/env bash
set -euo pipefail
./check_iface.sh eth0 || exit 1
./check_packet_loss.sh eth0 || exit 1
echo "Sanity test PASSED"
exit 0

#------------------------------------------------------------------

题目一：进程是否存在（pgrep）
🧪 面试题目

写一个脚本，检查指定进程是否正在运行

运行中 → 打印 OK，exit 0

未运行 → 打印错误，exit 1

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

proc_name="$1"

if [ -z "$proc_name" ]; then
    echo "Usage: $0 <process_name>"
    exit 1
fi

if pgrep "$proc_name" &>/dev/null; then
    echo "Process $proc_name is running"
    exit 0
else
    echo "Process $proc_name is NOT running"
    exit 1
fi

🔍 逐行解释
proc_name="$1"


👉 $1 是第一个参数
👉 存到语义化变量，方便后面用

if [ -z "$proc_name" ]; then


👉 -z 判断字符串是否为空
👉 判断用户有没有传参数

pgrep "$proc_name"


👉 查找进程名
👉 找到 → exit code = 0
👉 找不到 → exit code ≠ 0

&>/dev/null


👉 丢掉 stdout + stderr
👉 我们只关心 exit code

#-----------------------------------------------
如果 CPU 使用率 > 80%，脚本失败

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

cpu_idle=$(top -bn1 | awk '/Cpu/ {print $8}')
cpu_used=$(awk "BEGIN {print 100 - $cpu_idle}")

if [ "${cpu_used%.*}" -gt 80 ]; then
    echo "High CPU usage: $cpu_used%"
    exit 1
else
    echo "CPU usage OK: $cpu_used%"
    exit 0
fi

🔍 逐行解释
top -bn1


👉 -b 批处理
👉 -n1 只取一次

awk '/Cpu/ {print $8}'


👉 找包含 Cpu 的行
👉 $8 是 idle 百分比

cpu_used=$(awk "BEGIN {print 100 - $cpu_idle}")


👉 awk 计算数学表达式
👉 Shell 本身不擅长浮点

"${cpu_used%.*}"


👉 去掉小数部分
👉 方便做整数比较

#---------------------------------------------------
如果日志文件 5 分钟内没有更新 → 失败

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

log_file="$1"

if [ ! -f "$log_file" ]; then
    echo "Log file not found"
    exit 1
fi

if find "$log_file" -mmin +5 | grep -q .; then
    echo "Log is stale"
    exit 1
else
    echo "Log is updating"
    exit 0
fi

🔍 逐行解释
find "$log_file" -mmin +5


👉 文件修改时间 > 5 分钟

grep -q .


👉 只判断有没有输出
👉 有输出 → 文件老了

#---------------------------------------
题目四：参数解析（getopts）
🧪 面试题目

支持参数：
-i <interface>
-t <timeout>

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

while getopts "i:t:" opt; do
    case "$opt" in
        i) iface="$OPTARG" ;;
        t) timeout="$OPTARG" ;;
        *) exit 1 ;;
    esac
done

echo "Interface: $iface"
echo "Timeout: $timeout"

🔍 逐行解释
getopts "i:t:"


👉 i: 表示 -i 后面必须跟参数
👉 OPTARG 是参数值

case "$opt" in


👉 根据当前解析到的参数分支处理

#----------------------------------------------------
题目五：后台并发 + wait
🧪 面试题目

同时执行两个检测任务，全部完成后再继续

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

ping -c 3 8.8.8.8 &
pid1=$!

ping -c 3 1.1.1.1 &
pid2=$!

wait "$pid1" || exit 1
wait "$pid2" || exit 1

echo "All pings completed"
exit 0

🔍 逐行解释
&


👉 后台执行命令

$!


👉 最近一个后台进程的 PID

wait "$pid"


👉 等待指定进程
👉 返回该进程 exit code

#----------------------------------------------
题目六：root 权限检查
🧪 面试题目

如果不是 root 用户运行，直接失败

✅ 参考代码
#!/usr/bin/env bash
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

echo "Running as root"
exit 0

🔍 逐行解释
$EUID


👉 当前用户 ID
👉 root = 0

-ne


👉 数字不等于

#-------------------------------------------
一个服务在后台运行，会不断往日志里写内容。
你需要：

实时抓 log

一旦出现 ERROR 或 FATAL

立刻打印提示

脚本失败（exit 1）

如果 10 秒内没有错误

认为服务正常

exit 0

👉 这是 线上问题定位 / 自动化 sanity test 的典型场景。

✅ 参考脚本（完整）
#!/usr/bin/env bash
set -euo pipefail

log_file="$1"
timeout=10

if [ -z "$log_file" ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

if [ ! -f "$log_file" ]; then
    echo "Log file not found: $log_file"
    exit 1
fi

echo "Monitoring log for errors (timeout: ${timeout}s)..."

if timeout "$timeout" tail -F "$log_file" | grep -qE "ERROR|FATAL"; then
    echo "Error detected in log"
    exit 1
else
    echo "No error detected in ${timeout}s"
    exit 0
fi

🔍 逐行白话解释（重点来了）
第一行
#!/usr/bin/env bash


👉 明确告诉系统：这是 bash 脚本
👉 因为后面会用 bash 行为（管道 + timeout）

第二行
set -euo pipefail


-e：出错就停

-u：变量没定义就报错

pipefail：管道中任何一步失败都算失败

👉 对“自动化脚本”非常重要

参数接收
log_file="$1"
timeout=10


$1：第一个参数（日志文件路径）

timeout=10：监控 10 秒

👉 把“魔法数字”变成变量，是工程习惯

参数检查
if [ -z "$log_file" ]; then


👉 判断用户有没有传日志路径

if [ ! -f "$log_file" ]; then


👉 判断文件是否存在
👉 -f：普通文件
👉 !：取反

提示信息（给人看）
echo "Monitoring log for errors (timeout: ${timeout}s)..."


👉 纯提示，不影响逻辑
👉 在现场面试里这是加分点

核心逻辑（整题精华）
if timeout "$timeout" tail -F "$log_file" | grep -qE "ERROR|FATAL"; then


这一行我们拆成 4 层理解。

① tail -F "$log_file"

tail -F：

持续追踪文件

即使 log 被 rotate 也能继续

常用于抓运行中服务日志

👉 比 tail -f 更“工程化”

② timeout "$timeout" ...

最多运行 10 秒

超时后强制结束命令

👉 防止脚本永远卡住

③ 管道 |
tail -F ... | grep ...


tail 输出日志

grep 逐行检查

④ grep -qE "ERROR|FATAL"

-q：不打印任何内容

-E：支持扩展正则

"ERROR|FATAL"：匹配任意一个

👉 只关心：有没有出现过错误

if 判断靠的是什么？

不是看输出，而是看：

grep 的 exit code

找到匹配 → exit code = 0 → then

没找到 → exit code ≠ 0 → else

发现错误的情况
echo "Error detected in log"
exit 1


👉 给人看 + 给系统失败信号
👉 CI / 自动化立刻 stop

超时没发现错误
echo "No error detected in ${timeout}s"
exit 0


👉 表示 sanity test 通过

#-------------------------------------------------------
🧪 进阶题目：多日志 + 多关键字 + 并发抓 log
面试场景（非常真实）

一个系统有多个组件：

每个组件写自己的 log 文件

一旦任意 log 中出现 任意关键字

立即认为系统异常

脚本失败（exit 1）

同时要求：

并发监控多个 log

有超时保护（防止 hang）

🎯 功能要求总结

支持 多个 log 文件

支持 多个错误关键字（如 ERROR,FATAL,CRITICAL）

并发监控（每个 log 一个后台任务）

任一 log 触发错误 → 整体失败

超时未触发 → 整体成功

✅ 完整参考脚本（面试可写版本）
#!/usr/bin/env bash
set -euo pipefail

# ====== 参数配置 ======
timeout=10
keywords="ERROR|FATAL|CRITICAL"

# ====== 参数检查 ======
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <log_file1> [log_file2 ...]"
    exit 1
fi

# ====== 后台任务 PID 列表 ======
pids=()

# ====== 定义监控函数 ======
monitor_log() {
    local logfile="$1"

    if [ ! -f "$logfile" ]; then
        echo "Log file not found: $logfile"
        return 1
    fi

    echo "Monitoring $logfile"

    # 超时 + 实时抓 log + 关键字匹配
    if timeout "$timeout" tail -F "$logfile" | grep -qE "$keywords"; then
        echo "Error detected in $logfile"
        return 1
    fi

    return 0
}

# ====== 启动并发监控 ======
for logfile in "$@"; do
    monitor_log "$logfile" &
    pids+=($!)
done

# ====== 等待所有后台任务 ======
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "Sanity check FAILED"
        exit 1
    fi
done

echo "Sanity check PASSED"
exit 0

🔍 逐段 + 逐行详细解释（重点）
第一行：shebang
#!/usr/bin/env bash


👉 明确告诉系统：
这个脚本必须用 bash 跑

因为后面用了：

数组 pids=()

local

+=

第二行：安全模式
set -euo pipefail


-e：出错就停

-u：变量没定义就报错

pipefail：管道里任一步失败就失败

👉 自动化脚本必备

参数配置区
timeout=10
keywords="ERROR|FATAL|CRITICAL"


👉 把“可变配置”集中在一起

timeout：每个 log 监控多久

keywords：grep 的正则

📌 "A|B|C" = 任意一个匹配

参数数量检查
if [ "$#" -lt 1 ]; then


$#：参数个数

< 1：至少要一个 log 文件

echo "Usage: $0 <log_file1> [log_file2 ...]"
exit 1


👉 明确用法
👉 非 0 退出表示错误

PID 数组（并发核心）
pids=()


👉 bash 数组，用来存后台任务的 PID

定义监控函数
monitor_log() {
    local logfile="$1"


定义函数

local：变量只在函数里有效

文件存在性检查
if [ ! -f "$logfile" ]; then


👉 log 不存在直接失败

真正的“抓 log”逻辑（精华）
if timeout "$timeout" tail -F "$logfile" | grep -qE "$keywords"; then


这一行我们慢拆：

1️⃣ tail -F "$logfile"

实时跟踪 log

log rotate 也能继续

2️⃣ timeout "$timeout" ...

最多跑 N 秒

防止脚本卡死

3️⃣ | grep -qE "$keywords"

-q：不打印

-E：扩展正则

匹配任意关键字

📌 是否进入 if，完全由 exit code 决定

return vs exit（面试高频）
return 1


👉 只结束函数
👉 不结束整个脚本

并发启动（“多线程”的本质）
for logfile in "$@"; do
    monitor_log "$logfile" &
    pids+=($!)
done


逐行解释：

"$@"：所有参数（每个 log 文件）

&：后台执行（并发）

$!：刚启动的后台进程 PID

pids+=()：加入数组

📌 Shell 的并发 = 后台进程

等待所有后台任务
for pid in "${pids[@]}"; do


👉 遍历所有后台任务

if ! wait "$pid"; then


wait：等待某个 PID 结束

返回该任务的 exit code

!：只要失败就触发

exit 1


👉 任意一个 log 报错
👉 整体 sanity test 失败

全部通过
echo "Sanity check PASSED"
exit 0


👉 只有当 所有 log 都没出错 才会走到这一步