# ActaPinT 代码梳理与 Python 复现实验报告

## 1. 仓库概况

- 上游仓库：`https://github.com/wushulin/ActaPinT`
- 检查提交：`68431b7` (`main`)
- 原始内容：23 个 MATLAB `.m` 文件，共约 3,675 行
- 缺失内容：README、论文引用、依赖文件、测试、数据集、基准输出
- 计算类型：一维时变偏微分方程离散与时间并行算法，主要使用 CPU

服务器为 Tesla T4，但这些脚本的主要开销是稀疏线性代数、FFT 和小型
Newton/GMRES 迭代；原代码没有 CUDA/GPU 实现，因此复现使用 CPU 上的
NumPy/SciPy。

## 2. 代码结构

### 方程与参考解

- `Solution_Heat_ADE.m`：热方程/对流扩散方程，有限差分 + IMEX/θ 格式
- `Solution_Bergers.m`：黏性 Burgers 方程，Crank–Nicolson + Newton
- `Solution_Wave.m`：二阶波动方程转一阶系统，梯形格式
- `g.m`：源项函数

### 时间并行与全时域算法

- `Parareal_Heat_ADE.m`, `Parareal_Burgers.m`：Parareal
- `MGRiT_Heat_ADE.m`, `MGRiT_Burgers.m`：Parareal 与两层 MGRiT 对比
- `Direct_ParaDiag_*.m`：时间矩阵直接对角化
- `Iterative_ParaDiag_*.m`：α-circulant/FFT 预条件 GMRES 或定常迭代
- `Iterative_ParaDiag_RoundError.m`：ParaDiag-II 舍入误差实验
- `PCGC_Parareal_*.m`, `DiagG_Parareal_Heat_Wave.m`：对角化粗网格校正
- `STMG_*.m`：空间-时间两网格方法

### 其他方法

- `ParaExp_Burgers.m`：ParaExp
- `IDC_PIDC_ADE.m`：IDC/PIDC
- `SWR_for_ADE.m`：Schwarz waveform relaxation
- `UTP_DD_Method.m`, `RedBlackSubdomains.m`：波动方程时空区域分解

## 3. Python 重构

Python 代码位于：

- `actapint/operators.py`：周期/Dirichlet 一阶差分、负 Laplace、ADE 与
  Burgers 离散算子
- `actapint/integrators.py`：隐式 Euler、梯形、SDIRK22/23、Burgers
  Newton 求解
- `actapint/experiments.py`：可复现实验及绘图/指标输出
- `run_experiments.py`：统一命令行入口

已形成可运行 Python 实验：

1. Heat/ADE 参考解
2. Burgers 参考解
3. Wave 参考解
4. Heat/ADE Parareal
5. Burgers Parareal
6. Heat/ADE Parareal 与 MGRiT
7. ADE 迭代 ParaDiag（FFT 预条件 GMRES）
8. Heat/ADE 两网格 STMG 阻尼参数实验

这些实验覆盖仓库的三类模型方程和四条主要算法主线。其余 MATLAB 文件是
直接 ParaDiag、PCGC、ParaExp、IDC/PIDC、SWR 与区域分解等独立变体，
本轮已完成代码梳理，但未逐文件进行 Python 等价实现。

## 4. 原仓库问题与处理

`MGRiT_Heat_ADE.m` 第 13 行是无效 MATLAB：

```matlab
nu=0.002max;
```

Python 复现使用 `nu=0.002`。这是与周围 `nu>0` 分支、图标题和参数数量级
一致的最小修正。

原代码多处依赖 MATLAB `global`、脚本尾部局部函数和复制粘贴代码。Python
版本将算子、积分器和实验配置分离，以便测试和复用。

## 5. 原始规模复现结果

| 实验 | 结果 |
|---|---:|
| Heat/ADE Parareal | 39 次迭代，最终误差 `5.940e-15` |
| Burgers Parareal | 16 次迭代，最终误差 `3.577e-12` |
| Iterative ParaDiag ADE | GMRES 13 次，真实相对残差 `1.149e-14` |
| MGRiT ADE | 最终误差 `4.441e-16` |
| MGRiT 文件中的 Parareal（25 次限制） | 最终误差 `2.895e+02` |
| STMG ADE | `k=15` 最优采样阻尼约 `η=0.980`，误差 `1.494e-05` |
| Burgers 参考解 | 末时刻 `L∞=0.325871` |
| Wave 参考解 | 末时刻 `L∞=0.948217` |

MGRiT 对比脚本中的 Parareal 在前 25 次迭代放大并不代表实现崩溃：原脚本
本身把图的误差范围设为 `1e-2` 到 `1e4`，而 MGRiT 在相同设置下约 20 次
达到机器精度，正是该实验要展示的差异。

## 6. 运行方式

快速验证：

```bash
cd /root/ActaPinT
python3 run_experiments.py all --quick --output-dir results/quick
```

原始规模：

```bash
cd /root/ActaPinT
python3 run_experiments.py all --output-dir results/full
```

每个实验产生一张 PNG 图和一个 JSON 指标文件。

## 7. 可复现性说明

- 服务器未安装 MATLAB/Octave，且上游没有保存参考日志，因此无法做
  MATLAB 与 Python 的逐数值自动对拍。
- MATLAB `rng('default')` 与 NumPy 随机数生成器不同，随机全时域初值会使
  第一次误差不同；收敛终值和趋势更有比较意义。
- 稀疏 LU、FFT 和 GMRES 的舍入顺序不同，在 `1e-14` 到 `1e-16` 附近存在
  正常的平台差异。

