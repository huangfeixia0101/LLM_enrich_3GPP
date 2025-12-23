🧪 C 面试题 ①：抓 log，检测 ERROR（和你前面一致）
面试题目（非常真实）

用 C 语言 写一个程序：

接收一个参数：log 文件路径

逐行读取 log

如果发现 ERROR：

打印该行

程序失败（exit code = 1）

如果读完没有 ERROR：

程序成功（exit code = 0）

✅ 完整参考代码（现场可写，工程级）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    FILE *fp;
    char line[1024];

    // 1. 参数检查
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <log_file>\n", argv[0]);
        return 1;
    }

    // 2. 打开文件
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("Failed to open log file");
        return 1;
    }

    // 3. 逐行读取
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "ERROR") != NULL) {
            printf("Error detected: %s", line);
            fclose(fp);
            return 1;
        }
    }

    // 4. 正常结束
    fclose(fp);
    printf("No error found in log\n");
    return 0;
}

===============================================================
🧪 C 面试题：使用 popen() 调用系统命令并解析输出
面试场景（非常真实）

用 C 语言 写一个程序：

调用系统命令：ip link show eth0

读取命令输出

判断接口是否为 UP

如果是 UP：

打印 OK

exit 0

否则：

打印原因

exit 1

👉 这正是 DPU / 网络测试 / 系统 sanity check 的日常。

✅ 完整参考代码（面试可写，工程级）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    FILE *fp;
    char buffer[1024];
    int is_up = 0;

    // 1. 使用 popen 调用系统命令
    fp = popen("ip link show eth0", "r");
    if (fp == NULL) {
        perror("popen failed");
        return 1;
    }

    // 2. 逐行读取命令输出
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strstr(buffer, "UP") != NULL) {
            is_up = 1;
            break;
        }
    }

    // 3. 关闭管道，获取命令退出状态
    int status = pclose(fp);
    if (status == -1) {
        perror("pclose failed");
        return 1;
    }

    // 4. 根据解析结果判断
    if (is_up) {
        printf("Interface eth0 is UP\n");
        return 0;
    } else {
        printf("Interface eth0 is DOWN or not found\n");
        return 1;
    }
}

🔍 逐行、逐块白话解释（这是面试核心）

下面这一段，你可以当成面试时的“标准解说词”。

头文件
#include <stdio.h>


👉 提供：

FILE

fgets

printf

popen / pclose

#include <stdlib.h>


👉 提供：

perror

exit / return

#include <string.h>


👉 提供：

strstr（字符串匹配）

main 函数
int main(void)


👉 不需要参数
👉 命令写死为 eth0（面试 OK，真实工程可参数化）

buffer 定义（非常重要）
char buffer[1024];


👉 存放 命令输出的一行

📌 为什么是数组而不是指针？

避免未分配内存

防止段错误

调用系统命令（核心）
fp = popen("ip link show eth0", "r");

popen() 是什么？

在 C 程序里：

启动一个 shell 命令

返回一个 文件指针

可以像读文件一样读命令输出

📌 "r"：

只读命令的 stdout

popen 错误检查（必考）
if (fp == NULL)


👉 popen 失败（fork / shell 失败）
👉 必须检查

perror("popen failed");


👉 自动打印系统错误原因
👉 面试加分点

逐行读取命令输出
while (fgets(buffer, sizeof(buffer), fp) != NULL)


👉 和读 log 一样：

一行一行读

不加载全部输出

解析输出（关键逻辑）
if (strstr(buffer, "UP") != NULL)


👉 判断这一行里是否包含 UP

📌 在 ip link show 输出中：

<...UP,LOWER_UP...>

is_up = 1;
break;


👉 找到就够了
👉 不再继续读（效率 + 工程意识）

pclose（极其重要，面试高频）
int status = pclose(fp);

pclose() 做了什么？

关闭 pipe

等待子进程结束

返回 命令的退出状态

📌 和 fclose 不一样

pclose 错误检查
if (status == -1)


👉 说明 wait 失败
👉 这是老手才会写的检查

最终判断（exit code 设计）
if (is_up) {
    return 0;
} else {
    return 1;
}


👉 0 = 成功，非 0 = 失败
👉 和你前面的 Shell / Python 完全一致

====================================================================
🧪 C 面试题：实现「超时 + 重试」的系统检查
面试场景（100%真实）

写一个 C 程序：

检查某个系统条件（例如：网络接口是否 UP）

如果失败：

每隔 interval 秒重试

最多重试 max_retry 次

如果在超时前成功：

打印成功

exit 0

否则：

打印失败原因

exit 1

👉 这就是所有 sanity / bring-up / CI retry 的核心模型

✅ 完整参考代码（工程级，可现场写）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_RETRY 5
#define INTERVAL_SEC 2

int check_interface_up(void) {
    FILE *fp;
    char buffer[256];

    fp = popen("ip link show eth0", "r");
    if (fp == NULL) {
        return -1;
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strstr(buffer, "UP") != NULL) {
            pclose(fp);
            return 0;
        }
    }

    pclose(fp);
    return -1;
}

int main(void) {
    int retry = 0;

    while (retry < MAX_RETRY) {
        printf("Attempt %d/%d...\n", retry + 1, MAX_RETRY);

        if (check_interface_up() == 0) {
            printf("Interface eth0 is UP\n");
            return 0;
        }

        retry++;
        sleep(INTERVAL_SEC);
    }

    printf("Timeout: interface eth0 is still DOWN\n");
    return 1;
}

🔍 逐行 + 逐块详细解释（这是面试核心）

下面这一部分，你可以直接当成面试时的讲解稿。

一、头文件（为什么是这些）
#include <stdio.h>


👉 文件操作、FILE、fgets、printf

#include <stdlib.h>


👉 popen / pclose

#include <string.h>


👉 strstr（字符串匹配）

#include <unistd.h>


👉 sleep()
👉 Linux / Unix 系统编程必备

二、宏定义（工程习惯）
#define MAX_RETRY 5
#define INTERVAL_SEC 2


👉 把策略参数集中管理
👉 面试官很喜欢这点

重试次数

重试间隔

三、check_interface_up（单次检测函数）
int check_interface_up(void)


👉 单一职责：

只做一次检查

不关心 retry / timeout

📌 职责拆分 = 高级工程思维

四、调用系统命令
fp = popen("ip link show eth0", "r");


👉 启动 shell 命令
👉 返回 stdout 的 FILE*

五、为什么返回 -1？
if (fp == NULL) {
    return -1;
}


👉 统一失败语义
👉 0 = 成功，非 0 = 失败

📌 和 Shell / Python exit code 思维一致

六、逐行读取输出
while (fgets(buffer, sizeof(buffer), fp) != NULL)


👉 防止一次性读太多
👉 不占大内存

七、解析输出
if (strstr(buffer, "UP") != NULL)


👉 找到 UP
👉 说明接口已启动

八、为什么立刻 pclose + return？
pclose(fp);
return 0;


👉 两个关键点：

释放资源

提前结束（效率）

📌 面试官非常看重这一点

九、main 函数：重试逻辑核心
int retry = 0;


👉 重试计数器

十、while 循环（重试框架）
while (retry < MAX_RETRY)


👉 上限明确
👉 防止死循环

十一、打印当前尝试（调试友好）
printf("Attempt %d/%d...\n", retry + 1, MAX_RETRY);


👉 CI log 可读
👉 工程习惯

十二、调用单次检查
if (check_interface_up() == 0)


👉 成功即退出
👉 不再重试

十三、sleep 的意义（重点）
sleep(INTERVAL_SEC);


👉 防止 busy loop
👉 给系统恢复时间

📌 这是系统 bring-up / 网络测试的核心思想

十四、超时失败
printf("Timeout: interface eth0 is still DOWN\n");
return 1;


👉 达到最大重试次数
👉 明确失败原因
👉 返回失败 exit code

===================================================================
🧪 C 面试题：保存 Debug Artifacts（stdout / stderr 到文件）
面试场景（非常真实）

用 C 写一个系统检查程序：

调用系统命令（例如 ip link show eth0）

如果检查失败：

保存命令的 stdout

保存命令的 stderr

文件名带时间戳

返回 exit code：

成功 → 0

失败 → 1

👉 这就是 CI / bring-up / post-mortem 的标配能力

✅ 完整参考代码（工程级，面试可写）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ARTIFACT_DIR "artifacts"
#define CMD "ip link show eth0"

void create_artifact_dir(void) {
    mkdir(ARTIFACT_DIR, 0755);
}

void get_timestamp(char *buf, size_t size) {
    time_t now = time(NULL);
    struct tm *tm = gmtime(&now);
    strftime(buf, size, "%Y%m%d_%H%M%S", tm);
}

int run_command_and_save(void) {
    char cmd[256];
    char timestamp[64];
    char stdout_file[256];
    char stderr_file[256];

    get_timestamp(timestamp, sizeof(timestamp));
    create_artifact_dir();

    snprintf(stdout_file, sizeof(stdout_file),
             "%s/stdout_%s.txt", ARTIFACT_DIR, timestamp);
    snprintf(stderr_file, sizeof(stderr_file),
             "%s/stderr_%s.txt", ARTIFACT_DIR, timestamp);

    /*
     * shell 重定向：
     * 1> stdout_file
     * 2> stderr_file
     */
    snprintf(cmd, sizeof(cmd),
             "%s 1>%s 2>%s",
             CMD, stdout_file, stderr_file);

    int rc = system(cmd);
    return rc;
}

int main(void) {
    int rc = run_command_and_save();

    if (rc == 0) {
        printf("Command succeeded\n");
        return 0;
    } else {
        printf("Command failed, artifacts saved under %s/\n", ARTIFACT_DIR);
        return 1;
    }
}

🔍 逐行 + 逐块详细解释（这是面试真正要听的）

下面这部分你完全可以在面试中直接照着说。

一、为什么要保存 stdout / stderr？

在 CI / 自动化里：

❌ 只知道 FAIL → 没用

✅ 有 stdout / stderr → 能复盘

👉 debug artifact 是测试工程师的基本功

二、头文件解释
#include <stdio.h>


👉 printf

#include <stdlib.h>


👉 system

#include <string.h>


👉 snprintf

#include <time.h>


👉 时间戳

#include <unistd.h>


👉 Unix API（mkdir）

三、为什么用宏定义？
#define ARTIFACT_DIR "artifacts"
#define CMD "ip link show eth0"


👉 集中管理配置
👉 面试官会认为你有工程习惯

四、创建 artifact 目录
void create_artifact_dir(void) {
    mkdir(ARTIFACT_DIR, 0755);
}


👉 0755：

owner 可写

其他可读

📌 CI 标准权限

五、生成时间戳（关键）
strftime(buf, size, "%Y%m%d_%H%M%S", tm);


👉 文件名友好
👉 防止覆盖
👉 可追溯

六、为什么不用 popen 直接读？

这是面试官非常可能问的点。

你这里用的是：

system("cmd 1>stdout 2>stderr");

为什么这是好方案？

popen 默认只能拿 stdout

stderr 很难直接分流

shell 重定向：

简单

稳定

可读

👉 测试工具优先稳定，而不是“炫技”

七、stdout / stderr 重定向解释（必考）
%s 1>%s 2>%s


👉 在 shell 里：

写法	含义
1>	重定向 stdout
2>	重定向 stderr

📌 stdout / stderr 分离保存 = 高级点

八、为什么用 snprintf？
snprintf(...)


👉 防止 buffer overflow
👉 比 sprintf 安全

📌 这是 C 面试的加分点

九、system() 返回值
int rc = system(cmd);


👉 0 → 命令成功
👉 非 0 → 命令失败

📌 和你前面 Shell / Python exit code 完全一致

十、main 中的 exit code 设计
return 0; // success
return 1; // failure


👉 直接给 CI 用
👉 非常系统测试导向

🧠 面试官 100% 会追问的问题（答案我直接给你）
Q1：为什么不用 popen + dup2？

你可以答：

Shell redirection is simpler and more reliable
for test utilities where performance is not critical.

（满分回答）

Q2：system 有安全问题吗？

你可以说：

Yes, but here the command is controlled and not user input.

Q3：如果想保存更多信息？

你可以补一句：

We can also save exit code, timestamp, and environment info.