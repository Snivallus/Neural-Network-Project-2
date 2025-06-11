import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from thop import profile
from torchsummary import summary
import io
import contextlib
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import os


def get_stats(Data_set):
    # 此函数用于计算给定数据集 (Data_set) 中所有图像的通道均值和标准差
    # Data_set 中的每个元素是一个二元组 (image_tensor, label)
    # 其中 image_tensor 的形状为 [C, H, W], label 是该图像对应的类别标签

    # 通过列表推导式, 将数据集中所有样本的图像 Tensor 提取出来
    # item[0] 是图像 Tensor, item[1] 是标签, 故只取 item[0]
    imgs = [item[0] for item in Data_set]

    # torch.stack 会沿着维度 (dim=0) 将列表中的 Tensor 连接成一个更高维度的 Tensor
    # 假设 Data_set 中有 N 张图像, 每张图像的 Tensor 形状是 [3, 32, 32]
    # 那么执行 torch.stack(imgs, dim=0) 后得到的 Tensor 形状就是 [N, 3, 32, 32]
    imgs = torch.stack(imgs, dim=0)

    # 计算每个通道的均值和标准差
    means = imgs.mean(dim=(0, 2, 3))
    stds  = imgs.std(dim=(0, 2, 3))

    return means, stds

# PyTorch data loaders
def to_device(batch, device):
    # Moving tensors to the default device
    if isinstance(batch, (list,tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device, non_blocking=True)

class DeviceDataLoader():
    # Wrap a dataloader in order to transfer data to a device
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        # generating a batch of data after transferring data to the device
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        # batches size
        return len(self.dataloader)

    def __getattr__(self, name):
        # 如果有人写 train_loader.batch_size, 就可以直接返回底层 dataloader.batch_size
        return getattr(self.dataloader, name)
    
def imshow(imgs, means, stds):
    # 确保 Tensor 在转换为 NumPy 前已经位于 CPU 上
    # 从 PyTorch [C, H, W] 转换为 NumPy [H, W, C]
    imgs = imgs.cpu().numpy().transpose((1, 2, 0))

    # Unnormalize the image
    imgs = imgs * np.array(stds)[None, None, :] + np.array(means)[None, None, :]
    imgs = np.clip(imgs, 0, 1)  # Clip values to be in the range [0, 1] to avoid display issues

    plt.imshow(imgs)
    plt.show()

def show_batch_images(dataloader, classes, batch_size, mean, std):
    # Get a batch of training data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)  # use next() to fetch the next batch

    # Show images and labels
    imshow(torchvision.utils.make_grid(images), mean, std)
    # Display labels
    print('Labels:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(min(len(labels), batch_size))))


# 计算模型 (在一个批次中) 的准确率
def accuracy(outputs, labels):
    # torch.max 返回两个张量: 最大值和最大值对应的索引 (即预测的类别索引)
    # outputs 的形状通常是 [batch_size, num_classes],
    # 对 dim=1 取最大值就可以得到每行 (每个样本) 的预测类别
    _, predictions = torch.max(outputs, dim=1)
    # 计算并返回准确率
    correct = torch.sum(predictions == labels).float()
    return correct / labels.size(0)

# 基本的分类模型的训练和测试功能
class BaseClassification(nn.Module):
    # 处理单个训练批次, 返回字典包含该批次的损失和准确率
    def step(self, batch, train=True):
        # batch 是从 DataLoader 中得到的一对 (images, labels)
        images, labels = batch
        # 前向传播: 调用子类实现的 forward 方法, 对 images 得到输出 logits
        outputs = self(images)
        # 计算交叉熵损失
        loss = F.cross_entropy(outputs, labels)
        # 计算准确率
        acc = accuracy(outputs, labels)
        if train:
            return {'train_loss': loss, 'train_acc': acc}
        else:
            # 使用loss.detach()取消跟踪梯度, 以减少内存消耗
            return {'test_loss': loss.detach(), 'test_acc': acc}

    # 计算整个测试周期内的平均损失和平均准确率
    def epoch(self, results):
        batch_losses = [x['test_loss'] for x in results]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['test_acc'] for x in results]
        epoch_acc = sum(batch_accs) / len(batch_accs)
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item()}

    # 输出训练周期结束时的结果, 包括学习率、训练损失、训练准确率、验证损失和验证准确率
    def summarize_epoch(self, epoch, result):
        print("Epoch [{}], learning_rate: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(
            epoch,
            # result['lrs'] 是一个列表, 保存了每个迭代或每个 epoch 的学习率
            # 取最后一个元素, 表示当前 epoch 使用的学习率
            result['lrs'][-1],
            result['train_loss'],
            result['train_acc'],
            result['test_loss'],
            result['test_acc']
        ))

# 计算模型参数量的函数
def calc_param(model: nn.Module) -> int:
    # 将 model.parameters() 转换为 Python 列表, 包含所有可训练参数的 Tensor
    params = list(model.parameters())
    param_size = 0
    # 遍历每一个参数 Tensor
    for _param in params:
        _param_size = 1
        # 把该 Tensor 的所有维度大小相乘, 得到该 Tensor 总共有多少个标量元素
        for _dim in _param.size():
            _param_size *= _dim
        # 把当前参数的元素个数累加到总参数量
        param_size += _param_size
    # 返回所有可训练参数的总元素个数
    return param_size

def model_info(model, device):
    # 计算模型参数量
    print(f"模型参数量为: {calc_param(model)}")

    # 计算模型 FLOPs (浮点运算次数)
    # 创建一个 StringIO 对象来捕获输出
    buffer = io.StringIO()
    # 使用 redirect_stdout 将输出重定向到 buffer
    with contextlib.redirect_stdout(buffer):
        # 构造一个形状为 (1, 3, 32, 32) 的随机输入张量, 用于模拟一次前向传播
        # 这与 CIFAR-10 或类似小尺寸彩色图像的输入尺寸一致
        input = torch.randn(1, 3, 32, 32, device=device)
        # 调用 thop.profile, 计算给定模型在该输入下的 FLOPs 和参数量 (第二个返回值是 params，但我们这里不使用)
        flops, _ = profile(model, inputs=(input,))
    
    # 将 FLOPs 从“原始计数”转换为“十亿次浮点运算” (GFLOPs)
    gflops = flops / 1e9  # 将 FLOPs 转换为 GFLOPs
    print(f"FLOPs: {gflops:.3f} GFLOPs")

    # 计算模型所占空间大小
    # 创建一个 StringIO 对象来捕获输出
    buffer = io.StringIO()
    # 使用 redirect_stdout 将输出重定向到 buffer
    with contextlib.redirect_stdout(buffer):
      summary(model, (3, 32, 32))

    # 现在 buffer 包含了所有输出
    output = buffer.getvalue()
    # 关闭 buffer
    buffer.close()
    # 定义需要打印的行的开始标记
    keywords = [
        'Input size (MB)',
        'Forward/backward pass size (MB)',
        'Params size (MB)',
        'Estimated Total Size (MB)'
    ]
    # 遍历输出的每一行并打印包含关键字的行
    for line in output.split('\n'):
        # 如果该行包含我们关注的关键字之一, 就打印该行
        if any(keyword in line for keyword in keywords):
            print(line)

class Basic_Block(BaseClassification):
    def __init__(self, in_channels, out_channels, kernel_size, num, act_func="relu", pool=False, reduction=16):
        '''
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        num: 要重复卷积层的次数
        act_func: 激活函数类型: "relu" / "sigmoid" / "tanh"
        pool: 是否在卷积后添加池化层
        reduction (int): SE 模块中的通道压缩比，常设 16
        '''
        # 调用父类 BaseClassification 的构造函数
        super(Basic_Block, self).__init__()
    
        # ===========================
        # 1. 根据 act_func 参数，选择对应的激活函数
        #    支持 "relu" / "sigmoid" / "tanh"，默认为 ReLU
        self.act = self._get_activation(act_func)

        # ----------------------------
        # 2. 构建主体卷积序列 (Conv -> BN -> Act), 堆叠 num 层
        layers = []
        for i in range(num):
            # 对第一个卷积, 使用 pool 决定是否做下采样 (stride=2); 后续卷积 stride=1
            stride = 2 if (i == 0 and pool) else 1
            in_c = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            # 每层卷积后都使用相同类型的激活
            layers.append(self._get_activation(act_func))
        self.conv_sequence = nn.Sequential(*layers)

        # ----------------------------
        # 3. 如果需要下采样或通道对齐，则定义一个 downsample 分支 (1×1 Conv + BN)
        self.need_downsample = pool or (in_channels != out_channels)
        if self.need_downsample:
            ds_stride = 2 if pool else 1
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

        # ----------------------------
        # 4. Squeeze-and-Excitation 模块
        #    GAP -> FC (out_channels/reduction) -> Act (ReLU) -> FC (out_channels) -> Sigmoid
        #    最终输出与输入特征图逐通道相乘
        se_hidden = max(out_channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # [B, out_channels, 1, 1]
            nn.Conv2d(out_channels, se_hidden, 1), # [B, out_hidden, 1, 1]
            nn.ReLU(inplace=True),
            nn.Conv2d(se_hidden, out_channels, 1), # [B, out_channels, 1, 1]
            nn.Sigmoid()                           # [B, out_channels, 1, 1]
        )

    def _get_activation(self, act_func):
        """
        返回卷积层中使用的激活函数实例，与 act_func 保持一致。
        由于要在 ModuleList 的构造中调用，所以单独封装成方法。
        """
        if act_func.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif act_func.lower() == "sigmoid":
            return nn.Sigmoid()
        elif act_func.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

    # ===========================================
    # 前向传播方法
    def forward(self, x):
        """
        前向传播:
          1. x 可能先经过 downsample (如果 in_channels != out_channels 或 pool=True)
          2. conv_sequence 对 x 进行多层卷积->BN->激活 (第一层可能做 stride=2 下采样)
          3. 将 conv_sequence 的输出与 identity 相加，形成残差输出
          4. 对相加结果应用 SE 模块，做通道注意力加权
          5. 最后再经过一次激活函数
        """
        # 1. 保存恒等分支
        identity = x
        if self.need_downsample:
            identity = self.downsample(x)

        # 2. 主干序列卷积
        out = self.conv_sequence(x)
        # out 形状可能是 [B, out_channels, H/2, W/2] (如果下采样)，或 [B, out_channels, H, W]

        # 3. 残差相加
        out = out + identity

        # 4. SE 注意力: 先做 GAP -> FC -> ReLU -> FC -> Sigmoid -> 缩放通道
        se_weight = self.se(out)        # 形状 [B, out_channels, 1, 1]
        out = out * se_weight           # 通道级别加权
        return out
    
class CNN_model(BaseClassification):
    def __init__(self, act_func="relu", num_blocks=8):
        '''
        CIFAR_Block(in_channels, out_channels, kernel_size, num, pool)
          in_channels: 输入通道数
          out_channels: 输出通道数
          kernel_size: 卷积核大小
          num: 要重复卷积层的次数
          pool: 是否在卷积后添加池化层
        '''
        super(CNN_model, self).__init__()
        self.in_channels = 16  # 初始通道数
        self.act_func = act_func

        # 初始卷积: 输入为3通道，输出为16通道
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            self._get_activation(act_func)
        )

        # Stage 1: 输出通道数不变, 重复 num_blocks 个 block, 不下采样
        self.stage1 = self._make_stage(out_channels=16, num_blocks=num_blocks, pool_first=False)

        # Stage 2: 输出通道变为 32, 重复 num_blocks 个 block, 首个 block 下采样
        self.stage2 = self._make_stage(out_channels=32, num_blocks=num_blocks, pool_first=True)

        # Stage 3: 输出通道变为 64, 重复 num_blocks 个 block, 首个 block 下采样
        self.stage3 = self._make_stage(out_channels=64, num_blocks=num_blocks, pool_first=True)

        # 自适应平均池化层: 将特征图 [B, 128, H, W] 池化到 [B, 128, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc1=nn.Linear(64,128)
        # 激活函数
        self.act=self._get_activation(act_func)
        # 正则化, 随机丢弃一些激活值
        self.drop=nn.Dropout(0.5)
        # 全连接层
        self.fc2=nn.Linear(128,10)

        # ====================================
        # 4. 权重初始化：根据 act_func 选择合适的初始化方法
        self._initialize_weights(act_func)

    def _make_stage(self, out_channels, num_blocks, pool_first):
        """
        构建一个 stage, 由多个 Basic_Block_1 组成。
        - out_channels: 所有 block 的输出通道
        - num_blocks: block 的个数
        - pool_first: 是否第一个 block 做下采样
        """
        layers = []
        for i in range(num_blocks):
            pool = (i == 0 and pool_first)
            block = Basic_Block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=3,
                num=2,  # 每个 block 有 2 层 conv
                act_func=self.act_func,
                pool=pool
            )
            layers.append(block)
            self.in_channels = out_channels  # 下一层的输入通道对齐
        return nn.Sequential(*layers)

    def _get_activation(self, act_func):
        """
        返回卷积层中使用的激活函数实例, 与 act_func 保持一致.
        由于要在 ModuleList 的构造中调用，所以单独封装成方法.
        """
        if act_func.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif act_func.lower() == "sigmoid":
            return nn.Sigmoid()
        elif act_func.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

    def _initialize_weights(self, act_func):
        """
        对模型中所有可训练层 (Conv2d、Linear、BatchNorm) 进行初始化.
        - 若 act_func == "relu", 则对 Conv/Linear 使用 Kaiming Normal 初始化 (He 初始化).
        - 若 act_func == "sigmoid" 或 "tanh", 则对 Conv/Linear 使用 Xavier (Glorot) 初始化.
        - BatchNorm 的权重初始化为 1, 偏置初始化为 0.
        """
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                if act_func.lower() == "relu":
                    # Kaiming 正态初始化, 适用于 ReLU
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # Xavier 均匀初始化, 适用于 Sigmoid 和 Tanh
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 全连接层
            elif isinstance(m, nn.Linear):
                if act_func.lower() == "relu":
                    # Kaiming 正态初始化
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    # Xavier 均匀初始化
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # BatchNorm 层
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # 将所有 BatchNorm 的 scale 设置为 1，偏置设置为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 前向传播方法
    def forward(self, x):
        # 依次通过这六个卷积块进行特征提取
        output=self.stem(x)
        output=self.stage1(output)
        output=self.stage2(output)
        output=self.stage3(output)
        # 通过自适应平均池化层, 将每个特征图简化为一个单一的数值
        output=self.avgpool(output)
        # 将平坦化后的特征输入第一个全连接层
        output=self.fc1(output.view(output.size(0),-1))
        # 通过激活、dropout处理, 最后通过第二个全连接层得到最终的分类结果
        output=self.act(output)
        output=self.drop(output)
        output=self.fc2(output)
        return output

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        patience: 评估提前停止的周期数
        min_delta: 认为性能提升是显著的最小变化
        """
        # 保存用户设定的 patience 和 min_delta
        self.patience = patience
        self.min_delta = min_delta
        # best_acc 保存目前为止观察到的最高验证准确率
        self.best_acc = float('-inf')
        self.best_epoch = 0
        # wait 记录经过多少个 epoch 验证损失没有显著改善
        self.wait = 0
        # stopped_epoch 记录提前停止触发时的 epoch 编号
        self.stopped_epoch = 0
        # best_acc_changed 标记刚刚这个 epoch 是否更新了 best_acc
        self.best_acc_changed = False
        # early_stop 标记是否已经触发提前停止
        self.early_stop = False

    def __call__(self, test_acc, test_loss, epoch):
        # 比较当前准确率和历史最小准确率, 看是否有至少 min_delta 的上升
        # print(test_acc - self.best_acc) # debug
        if test_acc - self.best_acc > self.min_delta:
            # 1. 验证准确率显著下降
            self.best_acc = test_acc   # 更新最优验证准确率
            self.best_loss = test_loss # 更新最优验证损失
            self.best_epoch = epoch
            self.wait = 0              # 重置等待计数
            self.best_acc_changed = True
        else:
            # 2. 没有显著下降, 则将等待计数加 1
            self.wait += 1
            self.best_acc_changed = False
            # 如果超过或等于 patience, 则触发提前停止
            if self.wait >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                print(
                    f"Early stopping triggered after epoch {self.stopped_epoch}\n"
                    + f"Best accuracy {self.best_acc} occurred at epoch {self.best_epoch}"
                )

# 告诉PyTorch在执行evaluate函数时不需要计算梯度
@torch.no_grad()
def evaluate(model, test_loader):
    # 将模型设置为评估模式, BatchNorm/Dropout 等层将工作在推理状态
    model.eval()
    # 计算每个批次的损失和准确率
    device = next(model.parameters()).device
    outputs = []
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs.append(model.step((images, labels), train=False))
    # 计算并返回整个测试周期的平均损失和平均准确率
    return model.epoch(outputs)

# 从优化器中提取当前的基础学习率
def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

# 线性 Warmup
def warmup_lambda(epoch, warmup_epochs, total_epochs, min_lr_ratio=0.1):
    # 计算开始余弦退火的轮次
    max_lr_ratio = (total_epochs / warmup_epochs) / 5
    falling_epochs = warmup_epochs * (max_lr_ratio + 1)
    
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    elif epoch < falling_epochs:
        return 1.0
    else:
        # 余弦退火
        cosine_epoch = epoch - falling_epochs
        cosine_total = total_epochs - falling_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
# 使用 “Warmup + Adam” 学习率调度策略的训练循环
def fit_with_warmup_adam(
    epochs,
    max_lr,
    model,
    train_loader,
    test_loader,
    weight_decay=1e-4,
    grad_clip=None,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    warmup_epochs=5,
    best_model_path="./best_model.pth",
    patience=10,
    min_delta=0.001
):
    """
    训练模型：使用 Adam 优化器（自带每参数自适应学习率）+ 前 warmup_epochs 线性预热 + EarlyStopping。
    在 warmup 结束后, Adam 会凭借其内部一阶/二阶矩估计自动调节每个参数的实际更新步长。
    
    参数:
        epochs (int):           总的训练 epoch 数
        max_lr (float):         预热结束后的最大学习率，也是 Adam 的初始学习率
        model (nn.Module):      待训练的网络，继承 BaseClassification
        train_loader:           训练集 DataLoader
        test_loader:            测试集 DataLoader
        weight_decay (float):   L2 正则化系数 (默认 0)
        grad_clip (float|None): 梯度裁剪阈值 (None 表示不裁剪)
        beta1 (float):          Adam 的 β₁ 参数 (一阶矩衰减率)
        beta2 (float):          Adam 的 β₂ 参数 (二阶矩衰减率)
        eps (float):            Adam 的 ε 参数 (数值稳定项)
        warmup_epochs (int):    前 warmup 阶段的 epoch 数 (线性升到 max_lr)
        best_model_path (str):  保存最优模型的路径
        patience (int):         EarlyStopping 的耐心值
        min_delta (float):      EarlyStopping 判定“显著改善”的最小准确率差值
    
    返回:
        history (list of dict): 每个 epoch 结束后记录的结果列表，
                                每个元素包含 {'train_loss','train_acc','test_loss','test_acc','lrs'}
    """
    # 释放不再使用的显存
    torch.cuda.empty_cache()
    # 用来存储每个训练周期的结果
    history = []

    # ------------------------------
    # 1. 初始化优化器 (Adam)：
    #    在 warmup 期间，我们会把 lr 从 0 线性升到 max_lr
    optimizer = optim.Adam(
        model.parameters(),
        lr=max_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    # 2. 用 LambdaLR 实现 warmup
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_lambda(epoch, warmup_epochs, epochs)
    )

    # 3. 初始化 EarlyStopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # ------------------------------
    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        # 切换到训练模式 (启用 Dropout, BatchNorm 更新均值/方差)
        model.train()
        train_losses = []
        train_accs = []
        lrs = []

        # ------------------------------
        # 4.1 遍历所有训练批次
        for batch in train_loader:
            # batch = (images, labels)
            images, labels = batch
            images = images.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)

            # 前向 + loss + acc
            result = model.step((images, labels), train=True)
            loss = result['train_loss']        # Tensor 形式
            acc  = result['train_acc']         # Tensor 形式

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 可选梯度裁剪 (只在 warmup 完成后裁剪)
            if grad_clip is not None and epoch > warmup_epochs:
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # print(f"Grad norm before clipping: {total_norm:.4f}") # debug

            # 优化器更新参数
            optimizer.step()
            optimizer.zero_grad()

            # 记录训练指标
            train_losses.append(loss.detach())
            train_accs.append(acc)

            # 记录当前 "名义学习率"
            current_lr = get_learning_rate(optimizer)
            if not lrs:
                lrs.append(current_lr)

        # 4.2 更新学习率调度器 (按 epoch 更新)
        scheduler.step()

        # ------------------------------
        # 4.3 在测试集上进行评估 (evaluate 已自动关闭梯度)
        result = evaluate(model, test_loader)
        # 将训练集指标也合并到 result 中
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc']  = torch.stack(train_accs).mean().item()
        result['lrs'] = lrs

        # 打印本 epoch 的完整信息: lr, train_loss, train_acc, test_loss, test_acc
        model.summarize_epoch(epoch, result)

        # ------------------------------
        # 4.4 EarlyStopping 逻辑: 
        # 如果测试准确率连续多次没有显著下降, 就提前停止
        early_stopping(result['test_acc'], result['test_loss'], epoch)
        # 如果当前 epoch 使得 best_acc 更新了, 就保存最优模型
        if early_stopping.best_acc_changed:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': early_stopping.best_acc,
                'test_loss': early_stopping.best_loss
            }, best_model_path)
            print(
                f"New best model saved at epoch {epoch} with accuracy {early_stopping.best_acc:.4f}"
            )

        # ------------------------------
        # 将本 epoch 结果追加到 history，方便后续可视化或分析
        history.append(result)

        # 如果训练周期达到上限, 则输出最高准确率
        if epoch == epochs:
            print(f"Training completed after {epochs} epochs.")
            print(f"Max test accuracy: {early_stopping.best_acc:.4f} at epoch {early_stopping.best_epoch}")
            break

        # 如果触发了提前停止, 则结束训练
        if early_stopping.early_stop:
            print("Early stopping——training halted.\n")
            break

    return history

def fit_with_warmup_SGD(
    epochs,
    max_lr,
    model,
    train_loader,
    test_loader,
    weight_decay=1e-4,
    grad_clip=None,
    momentum=0.9,
    warmup_epochs=5,
    best_model_path="./best_model_sgd.pth",
    patience=10,
    min_delta=0.001
):
    """
    训练模型：使用 SGD 优化器 + 前 warmup_epochs 线性预热 + EarlyStopping。
    在 warmup 结束后, SGD 会以固定的学习率（或根据后续 Scheduler 的 lambda 函数）更新网络参数。
    通常在 SGD 中会配合 momentum, weight_decay 以提高收敛性能。

    参数:
        epochs (int):           总的训练 epoch 数
        max_lr (float):         预热结束后的最大学习率，也是 SGD 的初始学习率
        model (nn.Module):      待训练的网络，继承 BaseClassification 或 类似接口
        train_loader:           训练集 DataLoader
        test_loader:            测试集 DataLoader
        weight_decay (float):   L2 正则化系数
        grad_clip (float|None): 梯度裁剪阈值 (None 表示不裁剪)
        momentum (float):       SGD 中的动量系数
        warmup_epochs (int):    前 warmup 阶段的 epoch 数 (线性升到 max_lr)
        best_model_path (str):  保存最优模型的路径
        patience (int):         EarlyStopping 的耐心值
        min_delta (float):      EarlyStopping 判定“显著改善”的最小准确率差值

    返回:
        history (list of dict): 每个 epoch 结束后记录的结果列表，
                                每个元素包含 {'train_loss','train_acc','test_loss','test_acc','lrs'}
    """
    # ------------------------------
    # 0. 清理显存
    torch.cuda.empty_cache()

    # 1. 用来存储每个训练周期的结果
    history = []

    # ------------------------------
    # 2. 初始化优化器 (SGD)
    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 3. 用 LambdaLR 实现 warmup
    #    warmup_lambda 返回一个系数：当 epoch < warmup_epochs 时 linspace 从 0 到 1；
    #    epoch >= warmup_epochs 时 返回 1.0（保持 max_lr）
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_lambda(epoch, warmup_epochs, epochs)
    )

    # 4. 初始化 EarlyStopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # ------------------------------
    # 5. 训练循环
    for epoch in range(1, epochs + 1):
        # 切换到训练模式 (启用 Dropout, BatchNorm 更新均值/方差)
        model.train()
        train_losses = []
        train_accs = []
        lrs = []

        # 遍历所有训练批次
        for batch in train_loader:
            # batch = (images, labels)
            images, labels = batch
            # 将数据送到模型所在设备（如 GPU）
            device = next(model.parameters()).device
            images = images.to(device)
            labels = labels.to(device)

            # 前向 + 计算 loss、准确率（假设 model.step 接口与原来一致）
            result = model.step((images, labels), train=True)
            loss = result['train_loss']   # Tensor
            acc  = result['train_acc']    # Tensor

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (仅在 warmup 完成后裁剪)
            if grad_clip is not None and epoch > warmup_epochs:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 优化器更新参数（使用当前 lr 和动量）
            optimizer.step()
            optimizer.zero_grad()

            # 记录训练指标
            train_losses.append(loss.detach())
            train_accs.append(acc.detach())

            # 记录当前“名义学习率”（当前被 scheduler 应用后的 lr）
            current_lr = get_learning_rate(optimizer)
            if not lrs:
                lrs.append(current_lr)

        # 每个 epoch 结束后，更新学习率调度器（按 epoch 更新）
        scheduler.step()

        # ------------------------------
        # 在测试集上进行评估 (evaluate 已自动关闭梯度)
        result = evaluate(model, test_loader)
        # 合并训练集指标到 result 中
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc']  = torch.stack(train_accs).mean().item()
        result['lrs'] = lrs

        # 打印本 epoch 的完整信息: lr, train_loss, train_acc, test_loss, test_acc
        model.summarize_epoch(epoch, result)

        # ------------------------------
        # EarlyStopping 逻辑:
        early_stopping(result['test_acc'], result['test_loss'], epoch)
        # 如果当前 epoch 使得 best_acc 更新了, 保存最优模型
        if early_stopping.best_acc_changed:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': early_stopping.best_acc,
                'test_loss': early_stopping.best_loss
            }, best_model_path)
            print(f"New best model saved at epoch {epoch} with accuracy {early_stopping.best_acc:.4f}")

        # ------------------------------
        # 将本 epoch 结果追加到 history，方便后续可视化或分析
        history.append(result)

        # 如果训练周期达到上限, 则输出最高准确率
        if epoch == epochs:
            print(f"Training completed after {epochs} epochs.")
            print(f"Max test accuracy: {early_stopping.best_acc:.4f} at epoch {early_stopping.best_epoch}")
            break

        # 如果触发了提前停止，则结束训练
        if early_stopping.early_stop:
            print("Early stopping —— training halted.\n")
            break

    return history

# 绘制准确率曲线
def plot_accuracies(history):
    test_accuracies = [x['test_acc'] for x in history]
    train_accuracies = [x.get('train_acc') for x in history]
    plt.plot(train_accuracies, '-b')
    plt.plot(test_accuracies, '-r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

# 绘制对数损失曲线
def plot_log_loss(history):
    # 从历史记录中获取损失数据
    test_loss = [x['test_loss'] for x in history]
    train_loss = [x.get('train_loss') for x in history]
    train_loss[0] = 1
    # 计算损失的自然对数, 注意防止对非正数取对数
    
    log_train_loss = np.log(train_loss)
    log_test_loss = np.log(test_loss)
    # 绘制自然对数损失曲线
    plt.plot(log_train_loss, '-b')
    plt.plot(log_test_loss, '-r')
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    plt.legend(['Train', 'Test'])
    plt.title('Log Loss vs. No. of epochs')
    plt.show()

# 计算每一批次中模型在某一类别上的精确率
def precision_per_class(outputs, labels, class_index):
    _, predictions = torch.max(outputs, dim=1)
    true_positives = torch.sum((predictions == class_index) & (labels == class_index)).item()
    predicted_positives = torch.sum(predictions == class_index).item()
    return {'true_positives': true_positives, 'predicted_positives': predicted_positives}

def average_precision(model, test_loader, classes):
    num_classes = len(classes)
    precision_values = []
    # 初始化列表来存储每个类的真正例数、预测正例数
    class_true_positives = [0] * num_classes
    class_predicted_positives = [0] * num_classes

    # 遍历测试数据
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            outputs = model(images)
            for i in range(num_classes):
                result = precision_per_class(outputs, labels, i)
                class_true_positives[i] += result['true_positives']
                class_predicted_positives[i] += result['predicted_positives']

    # 计算每个类的平均精确率，并且取总的平均
    for i in range(num_classes):
        if class_predicted_positives[i] > 0:
            precision = class_true_positives[i] / class_predicted_positives[i]
            precision_values.append(precision)
            print("Precision of class {}: \t{:.4f}".format(classes[i], precision))
        else: # handle the case where class_predicted_positives[i] == 0
            print("Precision of class {}: \tN/A (no predicted positives)".format(classes[i]))

    # 计算所有类的平均精确率
    avg_precision_value = sum(precision_values) / len(precision_values)
    print("The Average precision value is {:.4f}\n".format(avg_precision_value))

# 1. Visualize Convolutional Filters (first layer)
def visualize_filters(model, layer_name='stem.0', num_filters=16, save_path='./figures/filter_visualization.pdf'):
    """
    Visualize the first num_filters convolutional filters in a 4x4 grayscale grid.

    Parameters:
        model: PyTorch model
        layer_name: string, path to the Conv2d layer (e.g., 'stem.0')
        num_filters: number of filters to visualize (default 16)
        save_path: path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Access the convolutional layer
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None or not isinstance(layer, nn.Conv2d):
        raise ValueError(f"Layer {layer_name} is not found or is not a Conv2d layer")

    # Get the filters: shape [out_channels, in_channels, k, k]
    weights = layer.weight.data.cpu().numpy()
    out_channels, in_channels, k, _ = weights.shape
    n_filters = min(num_filters, out_channels)
    
    # Set up the 4 x 4 grid
    n_rows, n_cols = 4, num_filters // 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()

    for i in range(n_filters):
        f = weights[i]  # shape: [in_channels, k, k]
        
        # Convert to grayscale representation:
        # Option 1: mean across channels
        f_gray = f.mean(axis=0)  # shape: [k, k]

        # Option 2 (alternative): use L2-norm: f_gray = np.sqrt(np.sum(f ** 2, axis=0))

        # Normalize to [0, 1]
        f_min, f_max = f_gray.min(), f_gray.max()
        if f_max - f_min > 1e-5:
            f_norm = (f_gray - f_min) / (f_max - f_min)
        else:
            f_norm = np.zeros_like(f_gray)

        axes[i].imshow(f_norm, cmap='gray')
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(n_filters, len(axes)):
        axes[j].axis('off')

    # plt.suptitle(f'Filters from layer {layer_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()

# 2. Plot Loss Landscape by linear interpolation between two sets of weights
def interpolate_loss(model, criterion, loader, state_dict_A, state_dict_B, steps=20):
    """
    For each alpha in [0,1], interpolate between two saved parameter states A and B:
      params = (1-alpha)*A + alpha*B
    Compute loss on one batch (for speed) and plot alpha vs loss.
    """
    # Collect parameters as flattened vectors
    params_A = torch.cat([v.view(-1) for v in state_dict_A.values()])
    params_B = torch.cat([v.view(-1) for v in state_dict_B.values()])
    
    alphas = np.linspace(0, 1, steps)
    losses = []
    
    device = next(model.parameters()).device
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    for alpha in alphas:
        # Build new state dict
        new_params = (1 - alpha) * params_A + alpha * params_B
        # Assign back to model
        pointer = 0
        new_state = {}
        for name, v in state_dict_A.items():
            numel = v.numel()
            new_state[name] = new_params[pointer:pointer+numel].view_as(v).clone()
            pointer += numel
        model.load_state_dict(new_state, strict=False)
        outputs = model(images)
        loss = criterion(outputs, labels).item()
        losses.append(loss)
    
    plt.plot(alphas, losses)
    plt.xlabel('Alpha (Interpolation)')
    plt.ylabel('Loss')
    plt.title('Linear Interpolation Loss Landscape')
    plt.savefig('./figures/loss_landscape_2d.pdf', dpi=300)
    plt.show()

# Helper to get all model parameters as a single vector
def get_params_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# Helper to set all model parameters from a single vector
def set_params_from_vector(model, vector):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vector[pointer:pointer + numel].view_as(p))
        pointer += numel

# Normalize direction
def normalize_direction(direction):
    norm = direction.norm()
    return direction / (norm + 1e-10)

# Compute loss on a single batch
def compute_loss(model, criterion, loader, device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
    return loss.item()

# Main function: Plot 3D loss landscape
def plot_loss_landscape_3D(model, criterion, loader, steps=21, scale=1.0):
    device = next(model.parameters()).device

    # Base point in weight space
    base_params = get_params_vector(model).to(device)

    # Random directions
    dir_x = normalize_direction(torch.randn_like(base_params))
    dir_y = torch.randn_like(base_params)
    dir_y -= torch.dot(dir_y, dir_x) * dir_x  # Gram-Schmidt
    dir_y = normalize_direction(dir_y)

    alphas = np.linspace(-scale, scale, steps)
    betas = np.linspace(-scale, scale, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            new_params = base_params + alpha * dir_x + beta * dir_y
            set_params_from_vector(model, new_params)
            loss = compute_loss(model, criterion, loader, device)
            losses[i, j] = loss

    # Plotting
    X, Y = np.meshgrid(alphas, betas)
    Z = losses.T

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.3, antialiased=True)
    # ax.set_title('3D Loss Landscape')
    ax.set_xlabel('Alpha (Direction X)')
    ax.set_ylabel('Beta (Direction Y)')
    ax.set_zlabel('Loss')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig('./figures/loss_landscape_3d.pdf', dpi=300)
    plt.show()