# 神经网络 Project 2

**安装依赖:**

```powershell
python -m venv venv # 搭建虚拟环境
source venv/bin/activate # 激活虚拟环境
pip install -r requirements.txt # 安装依赖
```

**项目结构:**

```bash
|-- 21307140051_雍崔扬_PJ2.pdf # 实验报告
|-- Latex # 实验报告的 Latex 源码
|-- README.md # README 文件
|-- VGG_Loss_Landscape.py # 用于绘制 VGG-A 和 VGG-A-BN 损失景观对比图
|-- cnn_models.ipynb # 最基本的训练模型的 notebook, 其中 model_1 达到了 92.06% 的准确率
|-- data
|   |-- __init__.py
|   |-- cifar-10-batches-py
|   |-- cifar-10-python.tar.gz # CIFAR-10 数据集
|   `-- loaders.py
|-- different_activation.ipynb # 比较不同激活函数效果的 notebook
|-- different_loss.ipynb # 比较不同 L2 正则项系数效果的 notebook
|-- different_optimizer.ipynb # 比较不同优化器效果的 notebook
|-- different_structure.ipynb # 比较不同模型架构 (参数量) 的 notebook
|-- figures # 第一个任务相关的图片
|   |-- accuracy_different_activation.pdf # 不同激活函数的效果
|   |-- accuracy_different_loss.pdf # 不同 L2 正则项系数的效果
|   |-- accuracy_different_optimizer.pdf # 不同优化器效果
|   |-- accuracy_different_structure.pdf # 不同模型架构效果
|   |-- filter_visualization.pdf # 卷积核权重的可视化
|   |-- learning_rate.pdf # 基础学习率的自定义调度方式示意图
|   |-- loss_landscape_2d.pdf # 2D 损失景观
|   `-- loss_landscape_3d.pdf # 3D 损失景观
|-- models # 第二个任务相关的模型
|   |-- __init__.py
|   `-- vgg.py # 定义 VCG-A 和 VCG-A-BN
|-- requirements.txt # 记录了项目依赖
|-- saved_weights # 第一个任务中保存的模型权重
|   |-- best_results # `cnn_models.ipynb` 保存的 model 0,1 的权重
|   |   -- best_model_0.pth
|   |   `-- best_model_1.pth
|   |-- different_activation # `different_activation.ipynb` 保存的 model 0,1,2 的权重
|   |   |-- best_model_0.pth
|   |   |-- best_model_0_next.pth
|   |   |-- best_model_1.pth
|   |   `-- best_model_2.pth
|   |-- different_loss # `different_loss.ipynb` 保存的 model 0,1,2 的权重
|   |   |-- best_model_0.pth
|   |   |-- best_model_1.pth
|   |   `-- best_model_2.pth
|   |-- different_optimizer # `different_optimizer.ipynb` 保存的 model 0,1 的权重
|   |   |-- best_model_0.pth
|   |   `-- best_model_1.pth
|   `-- different_structure # `different_structure.ipynb` 保存的 model 0,1,2 的权重
|       |-- best_model_0.pth
|       |-- best_model_1.pth
|       `-- best_model_2.pth
|-- utils # 工具
|   |-- __init__.py
|   |-- helper_functions.py # 第一个任务的 notebook 使用的帮助函数, 包括模型定义和训练函数等
|   `-- nn.py
`-- vgg_results	# 第二个任务相关的输出结果
    |-- norm_vgg_lr_0.0005.txt
    |-- norm_vgg_lr_0.001.txt
    |-- norm_vgg_lr_0.002.txt
    |-- saved_models # `VGG_Loss_Landscape.py` 保存的模型文件
    |   |-- vanilla_vgg_lr_0.0001_best.pth
    |   |-- vanilla_vgg_lr_0.0005_best.pth
    |   |-- vanilla_vgg_lr_0.001_best.pth
    |   |-- vanilla_vgg_lr_0.002_best.pth
    |   |-- vgg_bn_lr_0.0005_best.pth
    |   |-- vgg_bn_lr_0.001_best.pth
    |   `-- vgg_bn_lr_0.002_best.pth
    |-- vanilla_vgg_lr_0.0001.txt
    |-- vanilla_vgg_lr_0.0005.txt
    |-- vanilla_vgg_lr_0.001.txt
    |-- vanilla_vgg_lr_0.002.txt
    `-- visualization
        |-- landscape_comparison_lr0.0005.pdf
        |-- landscape_comparison_lr0.001.pdf
        `-- landscape_comparison_lr.pdf
```

**复现指南:**

- 本项目使用单张 $\text{NVIDIA RTX A6000}$ 显卡

- 对于第一个任务 (在 $\text{CIFAR-10}$ 上训练一个分类模型)，只需运行对应的五个 notebook 文件即可.

- 对于第二个任务 (研究 Batch Normalization 的作用)，`VCG_Loss_Landscape.py` 的用法如下:

  ```bash
  usage: VGG_Loss_Landscape.py [-h] 
  							 [--epoch_count EPOCH_COUNT] 
  							 [--learning_rates LEARNING_RATES] 
  							 [--seed_value SEED_VALUE]
                               [--initial_skip INITIAL_SKIP] 
                               [--plot_density PLOT_DENSITY] 
                               [--save_models]
  ```

  例如执行以下命令可绘制学习率为 $0.0005$ 时 $\text{VCG-A}$ 和 $\text{VCG-A-BN}$ 的损失景观对比图:  
  (保存模型权重，并且采样密度为 $5$)

  ```bash
  python VGG_Loss_Landscape.py --learning_rates 0.0005 --plot_density 5 --save_models
  ```
