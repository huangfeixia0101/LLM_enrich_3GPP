"梯度训练"
# L=(wx-y)^2
# dL/dW= d(wx-y)^2/d(wx-y)*d(wx-y)/dw
# import torch
#
# # 假设我们要拟合 y = 2x 的关系
# x = torch.tensor([1.0, 2.0, 3.0])
# y_true = torch.tensor([2.0, 4.0, 6.0])
#
# # 初始化参数 w（权重）
# w = torch.tensor(3.0, requires_grad=True)
#
# learning_rate = 0.1
#
# for epoch in range(10):
#     # 前向传播
#     y_pred = w * x
#     loss = ((y_pred - y_true)**2).mean()
#
#     # 反向传播
#     loss.backward()
#
#     # 手动更新参数
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#
#     # 清零梯度（否则下一轮会累加）
#     w.grad.zero_()
#
#     print(f"Epoch {epoch}: w = {w.item():.4f}, loss = {loss.item():.4f}， wgrad= {w.grad}")

# 神经网络最简单的 PyTorch 示例，训练一个包含线性层 + ReLU激活的小网络，去拟合函数
# 𝑦
# =
# 2
# 𝑥
# +
# 3
# # y=2x+3。
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 1. 准备训练数据 (x, y = 2x + 3)
# x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# y_train = 2 * x_train + 3
#
# # 假设的验证集（简单模拟）
# x_val = torch.tensor([[5.0], [6.0]])
# y_val = 2 * x_val + 3
#
# # 2. 定义网络结构
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.linear = nn.Linear(1, 1)  # 输入1维，输出1维
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = torch.relu(x)  # 激活函数
#         return x
#
# # 创建模型
# model = SimpleNet()
#
# # 3. 损失函数 & 优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # 4. 训练过程
# for epoch in range(300):
#     model.train()  # 设置为训练模式
#     optimizer.zero_grad()
#     outputs = model(x_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 50 == 0:
#         print(f"[训练] Epoch {epoch + 1}: loss={loss.item():.4f}")
#         for name, param in model.named_parameters():
#             print(f"  {name} = {param.data.item():.4f}")
#
# # 5. 评估阶段（验证集）
# model.eval()
# with torch.no_grad():
#     val_outputs = model(x_val)
#     val_loss = criterion(val_outputs, y_val)
#     print(f"\n[验证] 验证集 loss: {val_loss.item():.4f}")
#     for i in range(len(x_val)):
#         print(f"  输入 {x_val[i].item()} -> 预测值: {val_outputs[i].item():.4f}，真实值: {y_val[i].item()}")
#
# # 6. 保存模型
# torch.save(model.state_dict(), "simple_model.pth")
# print("\n✅ 模型已保存为 simple_model.pth")
#
# # 7. 加载模型并预测新的输入
# new_model = SimpleNet()
# new_model.load_state_dict(torch.load("simple_model.pth"))
# new_model.eval()
#
# with torch.no_grad():
#     test_input = torch.tensor([[8.0]])
#     pred = new_model(test_input)
#     print(f"\n🧪 预测输入8时的输出: {pred.item():.4f}")




# 多头attention
import torch
import torch.nn.functional as F

# 假设输入 token 数量 seq_len=2，embedding维度 d_model=8，头数 h=2
seq_len = 2
d_model = 8
num_heads = 2
d_head = d_model // num_heads  # 每个头的维度

# 输入：2个token，每个8维
x = torch.randn(seq_len, d_model)  # shape: (2, 8)

# 线性变换得到 Q, K, V，假设权重都用单位矩阵方便理解（这里直接用x）
Q = x.clone()
K = x.clone()
V = x.clone()

# 拆分为多个头
# 把embedding维度拆成头数份，每份维度是d_head
Q_heads = Q.view(seq_len, num_heads, d_head)  # (2, 2, 4)
K_heads = K.view(seq_len, num_heads, d_head)
V_heads = V.view(seq_len, num_heads, d_head)

print("Q_heads shape:", Q_heads.shape)  # (2 tokens, 2 heads, 4 dims per head)

# 计算每个头的注意力（简化版，不做缩放和softmax）
attn_scores = torch.einsum('thd,Thd->htT', Q_heads, K_heads)
# 解释：t=token，h=head，d=dim，输出形状(头数, 查询token, 关键token)
print("attn_scores shape:", attn_scores.shape)

# 对V加权求和（简化演示，假设权重为全1）
output_heads = V_heads  # 假设权重为1，直接用V
print("output_heads shape:", output_heads.shape)

# 合并所有头的输出，拼接最后一维
output_concat = output_heads.reshape(seq_len, d_model)
print("拼接后 output_concat 形状:", output_concat.shape)
