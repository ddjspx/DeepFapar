# 基于无人机多光谱与点云数据的植被生物物理参数估算

## 1. 项目背景

在植被生态学研究中，叶面积指数（LAI）与截获光合有效辐射比例（fAPAR）是表征陆地生态系统碳水循环及能量交换的核心生物物理参数。长期以来，植被性状监测面临着地面实测（In-situ measurement）代表性有限与卫星遥感（Satellite remote sensing）空间粗糙化（Mixed pixel effect）之间的尺度脱节。低空无人机（UAV）遥感平台的兴起，为亚米级甚至厘米级尺度的植被精细表征提供了高通量、非破坏性的观测手段，有效地弥合了微观单株尺度与宏观景观尺度之间的观测鸿沟。

## 2. 技术路径

本研究采用多源传感器数据融合策略，集成无人机多光谱影像提供的光谱反射率特征与点云数据（Point-cloud）派生的三维结构参数。针对植被指数在覆盖度较高时的"敏感度饱和"（Saturation effect）问题，通过引入随机森林（Random Forest）与神经网络（Neural Networks）等非参数化机器学习算法，能够更有效地捕捉复杂冠层结构下的非线性响应关系，从而提升生物物理变量反演的鲁棒性与精确度。

本仓库实现了基于全连接神经网络（Feedforward Neural Network）的端到端回归预测工作流，利用气象、植被和土壤等 10 个生态因子估算 fAPAR。

## 3. 项目结构

```
DeepFapar/
├── estimation.py                      # 主入口：串联数据划分 → 训练 → 测试 → 预测全流程
├── config/
│   ├── splits_params.py               # 数据集划分与归一化参数配置
│   └── NN_params.py                   # 神经网络超参数与预测配置
├── processing/
│   ├── splits_processing.py           # DatasetProcessor：数据读取、Min-Max 归一化、数据集划分
│   └── NN_processing.py               # NNProcessor：模型构建、训练、自动寻优、测试、推理
├── utils/
│   ├── metrics.py                     # 评估指标（R², RMSE, RRMSE, MAE, Nash-Sutcliffe）
│   ├── example_dataset_train_val_test.xlsx  # 训练/验证/测试数据集
│   └── example_dataset_predict.xlsx         # 待预测数据集
├── outputs/
│   ├── output_splits/                 # 归一化后的数据集划分结果及 Min-Max 参数
│   └── trial1_NN/                     # 模型训练输出：权重、曲线、评估指标、预测结果
├── environment.yml                    # Conda 环境配置
├── pyproject.toml                     # 项目元数据
├── result.md                          # 实验结果与解读报告
└── README.md
```

## 4. 核心模块

### 4.1 DatasetProcessor (`processing/splits_processing.py`)

负责数据预处理流水线：

- 读取原始生态数据集（Excel 格式）
- 对指定特征进行 Min-Max 归一化，消除量纲差异
- 按照 70% / 15% / 15% 的比例严格划分训练集、验证集与测试集
- 导出划分结果与归一化参数，确保预测阶段可复用相同的缩放基准

### 4.2 NNProcessor (`processing/NN_processing.py`)

核心训练与推理引擎，实现完整的模型生命周期管理：

- **模型构建**：基于 Keras Sequential API 搭建多层全连接网络
- **逐轮保存**：每个 Epoch 结束后自动保存完整模型权重（`epoch_XX.keras`）
- **自动寻优**：训练结束后回溯验证集 Loss 曲线，锁定最低 Loss 对应的 Epoch 权重作为最优模型
- **独立评估**：在测试集上生成预测-实测散点图与误差指标
- **未知数据预测**：加载最优模型对新数据进行推理，输出预测结果

### 4.3 error_metrics (`utils/metrics.py`)

计算遥感与生态学领域常用的 7 项评估指标：

| 指标 | 公式 | 含义 |
|---|---|---|
| r | Pearson 相关系数 | 线性相关程度 |
| R² | r² | 决定系数，模型解释的变异性比例 |
| MSE | mean((obs - sim)²) | 均方误差 |
| RMSE | √MSE | 均方根误差 |
| RRMSE | RMSE / mean(obs) × 100% | 相对均方根误差 |
| MAE | mean(\|obs - sim\|) | 平均绝对误差 |
| NASH | 1 - Σ(obs-sim)² / Σ(obs-mean(obs))² | 纳什效率系数 |

## 5. 实验配置

### 5.1 数据集划分参数 (`config/splits_params.py`)

| 参数 | 配置值 | 说明 |
|---|---|---|
| `training_size` | 0.7 | 训练集占比 |
| `test_size` | 0.15 | 测试集占比（验证集同样为 0.15） |
| `normalize` | True | 启用 Min-Max 归一化 |
| `target` | `fapar` | 目标变量 |

### 5.2 输入特征（10 个）

| 特征 | 含义 | 来源 |
|---|---|---|
| `logpre` | 降水量（对数变换） | 气象 |
| `logvpd` | 饱和气压差（对数变换） | 气象 |
| `chi` | 叶片经济性状指数 | 植被 |
| `logppfd` | 光合有效辐射（对数变换） | 气象 |
| `gtmp` | 温度 | 气象 |
| `AI` | 干旱指数 | 气候 |
| `loggpp` | 总初级生产力（对数变换） | 植被 |
| `soc` | 土壤有机碳 | 土壤 |
| `logtp` | 全磷（对数变换） | 土壤 |
| `logtn` | 全氮（对数变换） | 土壤 |

### 5.3 神经网络超参数 (`config/NN_params.py`)

| 参数 | 配置值 | 说明 |
|---|---|---|
| `depth` | 2 | 隐藏层数 |
| `neurons` | (64, 32) | 各层神经元数 |
| `activation` | relu | 激活函数 |
| `optimizer` | adam | 优化器 |
| `epochs` | 100 | 训练轮数 |
| `loss` | mse | 损失函数 |

**网络架构：**

```
Input (10) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
```

## 6. 使用方法

### 6.1 环境安装

```bash
# 方式 1：Conda（推荐）
conda env create -f environment.yml
conda activate deepfapar
pip install .

# 方式 2：pip
pip install numpy pandas matplotlib scikit-learn tensorflow keras openpyxl
```

### 6.2 运行实验

```bash
cd DeepFapar
python estimation.py
```

执行后将依次完成：

1. **数据划分** — 加载原始数据集，Min-Max 归一化，按 70/15/15 划分并保存至 `outputs/output_splits/`
2. **模型训练** — 构建神经网络，逐轮保存权重，自动选择验证集最优 Epoch，输出 Loss/MAE 曲线
3. **测试评估** — 在测试集上评估最优模型，生成散点图与误差指标
4. **预测推理** — 对未知数据集进行 fAPAR 预测，保存结果至 `outputs/trial1_NN/outputs_trial1_NN.xlsx`

### 6.3 自定义实验

修改 `config/` 目录下的配置文件即可快速开展新实验：

- 调整 `splits_params.py` 中的特征列表、划分比例或数据源
- 修改 `NN_params.py` 中的网络深度、神经元数、训练轮数等超参数
- 将新的预测数据路径填入 `NN_predict` 配置项

## 7. 输出文件

| 路径 | 内容 |
|---|---|
| `outputs/output_splits/X_train.xlsx` | 训练集特征 |
| `outputs/output_splits/y_train.xlsx` | 训练集标签 |
| `outputs/output_splits/X_val.xlsx` | 验证集特征 |
| `outputs/output_splits/y_val.xlsx` | 验证集标签 |
| `outputs/output_splits/X_test.xlsx` | 测试集特征 |
| `outputs/output_splits/y_test.xlsx` | 测试集标签 |
| `outputs/output_splits/min_max_values.xlsx` | Min-Max 归一化参数 |
| `outputs/trial1_NN/trial1_NN_model.keras` | 最优模型权重 |
| `outputs/trial1_NN/trial1_NN_loss_LAI_NN.png` | Loss 收敛曲线 |
| `outputs/trial1_NN/trial1_NN_mae_LAI_NN.png` | MAE 收敛曲线 |
| `outputs/trial1_NN/trial1_NN_validation_metrics.xlsx` | 验证集评估指标 |
| `outputs/trial1_NN/test/trial1_NN_inference_nn.png` | 测试集预测 vs 实测散点图 |
| `outputs/trial1_NN/outputs_trial1_NN.xlsx` | 未知数据预测结果 |

## 8. 模型选择策略

采用**验证集最优 Epoch 自动选择**机制：

1. 训练过程中每轮保存完整模型权重（`epoch_XX.keras`）
2. 训练结束后自动回溯验证集 Loss 曲线，锁定 Loss 最低的 Epoch
3. 加载该 Epoch 的权重作为最终模型

该策略避免了手动设置早停阈值，确保模型在欠拟合与过拟合之间取得最佳平衡。

## 9. 数据可复现性

为遵循可重复研究（Reproducible Research）的学术规范，本项目构建了模块化的数据预处理与模型训练工作流：

- **参数化配置**：所有实验参数集中管理于 `config/` 目录，修改配置即可复现实验
- **标准化数据接口**：基于 Excel 格式的数据集，便于跨平台读取与校验
- **全流程透明化**：从特征工程、模型校准（Calibration）到验证（Validation）的每个步骤均有独立输出
- **可追溯性**：逐 Epoch 模型权重保存，支持任意中间状态的回溯与分析

实验结果与详细解读见仓库 `result.md` 文件。

## 10. 依赖

- Python >= 3.8, < 3.12
- numpy >= 1.23
- pandas >= 1.4
- matplotlib >= 3.5
- scikit-learn >= 1.1
- tensorflow >= 2.11
- keras >= 2.11
- openpyxl >= 3.0

## 11. 引用

数据仓库地址：https://github.com/Xczx1000/DeepFapar
