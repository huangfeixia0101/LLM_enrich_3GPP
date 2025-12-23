# =================================
# File : nvidia.py
# Description : 
# Author : Fei
# CREATE TIME : 2025/12/23 6:58
# =================================

🧪 Python 面试题 ①：调用 shell 检查网络接口
题目（非常真实）

用 Python 写一个脚本：

接收一个参数：网络接口名（如 eth0）

调用系统命令检查接口是否存在

如果存在且是 UP：

打印 OK

退出码 = 0

否则：

打印错误原因

退出码 ≠ 0

✅ 标准参考代码（面试可写）
#!/usr/bin/env python3

import subprocess
import sys

def check_interface(iface: str) -> int:
    try:
        result = subprocess.run(
            ["ip", "link", "show", iface],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Interface not found: {iface}")
        return 1

    if "UP" in result.stdout:
        print(f"Interface {iface} is UP")
        return 0
    else:
        print(f"Interface {iface} is DOWN")
        return 1

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <interface>")
        sys.exit(1)

    iface = sys.argv[1]
    rc = check_interface(iface)
    sys.exit(rc)

if __name__ == "__main__":
    main()

五、逐行“白话解释”（和 Shell 一样细）
shebang
#!/usr/bin/env python3


👉 告诉系统：
这是 Python 3 脚本，不是 Python 2

import
import subprocess
import sys


subprocess：跑 shell 命令

sys：处理参数 & exit code

核心函数
def check_interface(iface: str) -> int:


👉 定义一个函数
👉 输入：接口名
👉 输出：exit code（0 / 1）

📌 函数返回 exit code = 非常 SDET 的写法

subprocess.run（核心）
result = subprocess.run(
    ["ip", "link", "show", iface],


👉 不用 shell=True（安全）
👉 命令和参数分开写

stdout=subprocess.PIPE,
stderr=subprocess.PIPE,
text=True,


捕获输出

text=True：返回字符串而不是 bytes

check=True


👉 如果命令 exit code ≠ 0
👉 直接抛异常

异常处理（面试加分点）
except subprocess.CalledProcessError:


👉 接口不存在 / 命令失败
👉 用异常处理，而不是 if 判断

判断 UP / DOWN
if "UP" in result.stdout:


👉 简单、直观
👉 不玩正则（除非必要）

main 函数（工程感）
if __name__ == "__main__":


👉 脚本模式入口
👉 可测试、可复用

exit code
sys.exit(rc)


👉 Python 脚本的 exit code = rc

0：成功

非 0：失败

👉 和 shell / CI 完美对齐

# ===========================================================
🧪 Python 面试题：实时抓 log + 自动判错
面试场景（非常真实）

一个后台服务在运行，不断往 log 文件写内容。
你需要用 Python：

实时监控 log 文件

如果出现关键字（如 ERROR / FATAL）：

立即打印提示

脚本失败（exit code ≠ 0）

如果在 指定时间内 没有错误：

认为服务正常

exit code = 0

👉 这就是 sanity test / health check / CI pre-check。

✅ 参考代码（面试可接受版本）
#!/usr/bin/env python3

import sys
import time
from typing import List

def monitor_log(
    log_file: str,
    keywords: List[str],
    timeout: int
) -> int:
    """
    Monitor log file for given keywords within timeout.
    Return 0 if no error found, 1 if error detected.
    """

    try:
        with open(log_file, "r") as f:
            # 移动到文件末尾，只看新日志
            f.seek(0, 2)

            start_time = time.time()

            while True:
                line = f.readline()

                if line:
                    for kw in keywords:
                        if kw in line:
                            print(f"Error detected: '{kw}' in log")
                            print(line.strip())
                            return 1
                else:
                    # 没有新日志，避免 busy loop
                    time.sleep(0.2)

                # 超时判断
                if time.time() - start_time > timeout:
                    return 0

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return 1

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    keywords = ["ERROR", "FATAL"]
    timeout = 10

    rc = monitor_log(log_file, keywords, timeout)
    sys.exit(rc)

if __name__ == "__main__":
    main()

🔍 逐行 + 逐块详细解释（重点）

下面开始 逐句白话解释，你可以照着这个给面试官讲。

第一行：shebang
#!/usr/bin/env python3


👉 告诉系统：
用 Python 3 来执行这个脚本

避免 Python 2

和 Shell 的 shebang 是同一个逻辑

import 部分
import sys
import time
from typing import List

为什么是这些模块？

sys：拿参数、设置 exit code

time：超时、sleep

List：类型提示（加分项）

📌 类型提示不是必须，但在 Senior / SDET 面试里是加分的。

核心函数定义
def monitor_log(
    log_file: str,
    keywords: List[str],
    timeout: int
) -> int:


👉 定义一个函数来做“抓 log”

输入：

log_file：日志路径

keywords：错误关键字列表

timeout：监控时长

输出：

int：exit code（0 / 1）

📌 函数返回 exit code 是非常典型的测试工程师写法。

打开文件（关键）
with open(log_file, "r") as f:


👉 只读方式打开 log 文件
👉 with：自动关闭文件（工程习惯）

移动到文件末尾
f.seek(0, 2)


这一行非常重要。

含义是：

0：偏移量

2：从文件末尾开始

👉 等价于：

“不看历史 log，只看新写入的 log”

📌 和 tail -f 的行为一致。

记录开始时间
start_time = time.time()


👉 当前时间戳（秒）
👉 用来做 timeout 判断

主循环（核心）
while True:


👉 一直循环，直到：

发现错误

或超时

读一行 log
line = f.readline()


如果有新日志 → 返回字符串

如果没新日志 → 返回空字符串 ""

如果读到了新日志
if line:


👉 有新内容

逐个关键字检查
for kw in keywords:
    if kw in line:


👉 简单直观：

不用正则

面试里更安全

发现错误
print(f"Error detected: '{kw}' in log")
print(line.strip())
return 1


👉 做三件事：

打印错误原因

打印具体 log 行

返回 1（失败）

📌 return 1 ≈ exit 1（但只退出函数）

没有新日志时
else:
    time.sleep(0.2)


👉 防止 CPU 空转（busy loop）
👉 这是 非常重要的工程细节

超时判断
if time.time() - start_time > timeout:
    return 0


👉 超过指定时间
👉 没有发现错误
👉 返回成功

异常处理
except FileNotFoundError:


👉 日志文件不存在
👉 明确打印错误
👉 返回失败

main 函数（工程结构）
def main():


👉 标准入口
👉 可测试、可复用

参数检查
if len(sys.argv) < 2:


👉 用户没传 log 文件路径

配置关键字 & 超时
keywords = ["ERROR", "FATAL"]
timeout = 10


👉 集中配置
👉 面试官会喜欢你这么写

exit code 统一出口
rc = monitor_log(...)
sys.exit(rc)


👉 Python 脚本的 exit code = rc

# =========================================================
🧪 Python 面试题：Test Runner（用 exit code 汇总多个 check）
面试背景（非常真实）

系统启动后，需要做一组 sanity checks：

检查网络接口

检查日志是否有错误

检查进程是否存活

要求：

每个 check 独立执行

每个 check 返回 exit code（0 / 非 0）

只要有一个失败

整体失败

最终 exit code ≠ 0

全部通过

exit code = 0

👉 这就是 CI / 自动化 / Sanity Test 的核心模式。

✅ 完整参考代码（现场可写版本）
#!/usr/bin/env python3

import sys
import subprocess
from typing import Callable, List, Tuple

# ===== 单个 check 示例 =====

def check_interface() -> int:
    """Check if eth0 is UP"""
    try:
        result = subprocess.run(
            ["ip", "link", "show", "eth0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError:
        print("[FAIL] Interface eth0 not found")
        return 1

    if "UP" in result.stdout:
        print("[PASS] Interface eth0 is UP")
        return 0
    else:
        print("[FAIL] Interface eth0 is DOWN")
        return 1


def check_process() -> int:
    """Check if sshd process is running"""
    result = subprocess.run(
        ["pgrep", "sshd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        print("[PASS] sshd process is running")
        return 0
    else:
        print("[FAIL] sshd process is NOT running")
        return 1


def check_log() -> int:
    """使用上面写的readline抓log的代码这里不重复了"""
    print("[PASS] No error found in log")
    return 0


# ===== Test Runner 核心 =====

def run_tests(tests: List[Tuple[str, Callable[[], int]]]) -> int:
    """
    Run all tests and aggregate results.
    Return 0 if all pass, otherwise return 1.
    """
    failed = False

    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        rc = test_func()

        if rc != 0:
            failed = True

    return 1 if failed else 0


def main():
    tests = [
        ("Check Interface", check_interface),
        ("Check Process", check_process),
        ("Check Log", check_log),
    ]

    final_rc = run_tests(tests)

    if final_rc == 0:
        print("\n=== SANITY TEST PASSED ===")
    else:
        print("\n=== SANITY TEST FAILED ===")

    sys.exit(final_rc)


if __name__ == "__main__":
    main()

# ====================================================================
🧪 面试题：Python Test Runner + JSON 报告输出
面试场景（非常真实）

写一个 Python 测试脚本：

执行多个 sanity checks

每个 check：

有名字

有 PASS / FAIL

有 message

最终生成一个 JSON 报告文件

同时：

所有 check PASS → exit 0

任一 FAIL → exit 1

👉 这就是 CI 自动化测试报告的标准形态。

✅ 完整参考代码（现场可写版）
#!/usr/bin/env python3

import sys
import subprocess
import json
import time
from typing import Callable, Dict, List


# ===== 单个 check 示例 =====

def check_interface() -> Dict:
    start = time.time()

    try:
        result = subprocess.run(
            ["ip", "link", "show", "eth0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if "UP" in result.stdout:
            status = "PASS"
            message = "Interface eth0 is UP"
        else:
            status = "FAIL"
            message = "Interface eth0 is DOWN"

    except subprocess.CalledProcessError:
        status = "FAIL"
        message = "Interface eth0 not found"

    duration = time.time() - start

    return {
        "name": "check_interface",
        "status": status,
        "message": message,
        "duration_sec": round(duration, 2)
    }


def check_process() -> Dict:
    start = time.time()

    result = subprocess.run(
        ["pgrep", "sshd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        status = "PASS"
        message = "sshd process is running"
    else:
        status = "FAIL"
        message = "sshd process is NOT running"

    duration = time.time() - start

    return {
        "name": "check_process",
        "status": status,
        "message": message,
        "duration_sec": round(duration, 2)
    }


# ===== Test Runner + JSON 汇总 =====

def run_tests(tests: List[Callable[[], Dict]]) -> Dict:
    results = []
    overall_status = "PASS"

    for test_func in tests:
        result = test_func()
        results.append(result)

        if result["status"] != "PASS":
            overall_status = "FAIL"

    return {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "overall_status": overall_status
        },
        "results": results
    }


def main():
    tests = [
        check_interface,
        check_process,
    ]

    report = run_tests(tests)

    with open("sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if report["summary"]["overall_status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

# ==========================================================================
我们就在你刚才那份 Python + JSON test runner 基础上，干净地加上：

✅ timestamp（什么时候跑的）

✅ hostname（在哪台机器 / 哪个节点跑的）

✅ env 信息（环境变量 / 测试环境）
#!/usr/bin/env python3

import sys
import subprocess
import json
import time
import socket
import os
from typing import Callable, Dict, List


# ===== 单个 check 示例 =====

def check_interface() -> Dict:
    start = time.time()

    try:
        result = subprocess.run(
            ["ip", "link", "show", "eth0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if "UP" in result.stdout:
            status = "PASS"
            message = "Interface eth0 is UP"
        else:
            status = "FAIL"
            message = "Interface eth0 is DOWN"

    except subprocess.CalledProcessError:
        status = "FAIL"
        message = "Interface eth0 not found"

    duration = time.time() - start

    return {
        "name": "check_interface",
        "status": status,
        "message": message,
        "duration_sec": round(duration, 2)
    }


def check_process() -> Dict:
    start = time.time()

    result = subprocess.run(
        ["pgrep", "sshd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        status = "PASS"
        message = "sshd process is running"
    else:
        status = "FAIL"
        message = "sshd process is NOT running"

    duration = time.time() - start

    return {
        "name": "check_process",
        "status": status,
        "message": message,
        "duration_sec": round(duration, 2)
    }


# ===== Test Runner + JSON 汇总 =====

def run_tests(tests: List[Callable[[], Dict]]) -> Dict:
    results = []
    overall_status = "PASS"

    for test_func in tests:
        result = test_func()
        results.append(result)

        if result["status"] != "PASS":
            overall_status = "FAIL"

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "overall_status": overall_status
    }

    # ===== 新增：meta 信息 =====
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "env": {
            "TEST_ENV": os.getenv("TEST_ENV", "unknown"),
            "BUILD_ID": os.getenv("BUILD_ID", "unknown")
        }
    }

    return {
        "meta": meta,
        "summary": summary,
        "results": results
    }


def main():
    tests = [
        check_interface,
        check_process,
    ]

    report = run_tests(tests)

    with open("sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if report["summary"]["overall_status"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

# =================================================================
🧪 升级题目：失败时自动保存 Debug Artifacts
真实工程场景

在 CI / DPU / 系统测试里：

FAIL 本身 不值钱

为什么 FAIL 才值钱

所以要求：

如果某个 check FAIL

自动保存：

命令 stdout

命令 stderr

相关 log 文件

保存到：

带时间戳的目录

JSON 报告里：

记录 artifact 路径

🧪 升级题目：失败时自动保存 Debug Artifacts
真实工程场景

在 CI / DPU / 系统测试里：

FAIL 本身 不值钱

为什么 FAIL 才值钱

所以要求：

如果某个 check FAIL

自动保存：

命令 stdout

命令 stderr

相关 log 文件

保存到：

带时间戳的目录

JSON 报告里：

记录 artifact 路径

✅ 完整参考代码（CI 级，面试可写）
#!/usr/bin/env python3

import sys
import subprocess
import json
import time
import socket
import os
from typing import Callable, Dict, List


ARTIFACT_DIR = "artifacts"


# ===== 工具函数：创建 artifact 目录 =====

def create_artifact_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    path = os.path.join(ARTIFACT_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def save_text(path: str, filename: str, content: str) -> None:
    with open(os.path.join(path, filename), "w") as f:
        f.write(content)


# ===== 单个 check：带 debug artifact =====

def check_interface() -> Dict:
    start = time.time()
    artifact_path = None

    try:
        result = subprocess.run(
            ["ip", "link", "show", "eth0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if "UP" in result.stdout:
            status = "PASS"
            message = "Interface eth0 is UP"
        else:
            status = "FAIL"
            message = "Interface eth0 is DOWN"

    except subprocess.CalledProcessError as e:
        status = "FAIL"
        message = "Interface eth0 not found"
        result = e

    duration = time.time() - start

    # ===== 失败时保存 artifact =====
    if status == "FAIL":
        artifact_path = create_artifact_dir()
        save_text(artifact_path, "stdout.txt", result.stdout or "")
        save_text(artifact_path, "stderr.txt", result.stderr or "")

    return {
        "name": "check_interface",
        "status": status,
        "message": message,
        "duration_sec": round(duration, 2),
        "artifact_path": artifact_path
    }


# ===== Test Runner =====

def run_tests(tests: List[Callable[[], Dict]]) -> Dict:
    results = []
    overall_status = "PASS"

    for test_func in tests:
        result = test_func()
        results.append(result)

        if result["status"] != "PASS":
            overall_status = "FAIL"

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "env": {
            "TEST_ENV": os.getenv("TEST_ENV", "unknown"),
            "BUILD_ID": os.getenv("BUILD_ID", "unknown")
        }
    }

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "overall_status": overall_status
    }

    return {
        "meta": meta,
        "summary": summary,
        "results": results
    }


def main():
    tests = [
        check_interface,
    ]

    report = run_tests(tests)

    with open("sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    sys.exit(0 if report["summary"]["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
