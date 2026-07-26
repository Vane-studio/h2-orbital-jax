# H₂ Orbital JAX

> 使用 JAX、LCAO 与 Marching Cubes，从氢原子波函数构造并可视化氢分子的成键与反键分子轨道。

## 项目简介

`H₂ Orbital JAX` 是一个面向计算物理、量子化学与科学可视化的实验性项目。

项目从氢原子归一化 $1s$ 波函数出发，使用原子轨道线性组合方法（Linear Combination of Atomic Orbitals, LCAO）构造氢分子的成键轨道与反键轨道，并在三维笛卡尔网格上完成：

- 原子轨道与分子轨道计算；
- 波函数归一化与正交性验证；
- 概率密度离散化；
- 累计概率等值面求解；
- Marching Cubes 三角网格提取；
- Plotly 交互式三维可视化；
- OBJ、PLY 与 GLB 模型导出；
- 实验参数、软件版本和数值结果记录。

项目核心映射为：

$$
\psi(\mathbf r)
\longrightarrow
|\psi(\mathbf r)|^2
\longrightarrow
\mathbf T
\longrightarrow
\text{Isosurface}
\longrightarrow
\text{Triangle Mesh}.
$$

---

## 物理模型

两个氢原子核沿 $x$ 轴放置：

$$
\mathbf R_A=
\left(-\frac R2,0,0\right),
\qquad
\mathbf R_B=
\left(+\frac R2,0,0\right).
$$

原子单位制下，归一化氢原子 $1s$ 波函数为：

$$
\phi_{1s}(r)
=
\frac{e^{-r}}{\sqrt{\pi}}.
$$

两个原子轨道的重叠积分为：

$$
S(R)
=
e^{-R}
\left(
1+R+\frac{R^2}{3}
\right).
$$

成键轨道为：

$$
\psi_{\sigma}
=
\frac{\phi_A+\phi_B}
{\sqrt{2(1+S)}}.
$$

反键轨道为：

$$
\psi_{\sigma^\ast}
=
\frac{\phi_A-\phi_B}
{\sqrt{2(1-S)}}.
$$

对应概率密度为：

$$
\rho(\mathbf r)
=
|\psi(\mathbf r)|^2.
$$

---

## 数值验证

程序会自动检查：

$$
\langle\phi_A|\phi_A\rangle\approx1,
$$

$$
\langle\phi_B|\phi_B\rangle\approx1,
$$

$$
\langle\psi_\sigma|\psi_\sigma\rangle\approx1,
$$

$$
\langle\psi_{\sigma^\ast}|\psi_{\sigma^\ast}\rangle\approx1,
$$

以及

$$
\langle
\psi_\sigma
|
\psi_{\sigma^\ast}
\rangle
\approx0.
$$

在默认参数 $R=1.4a_0$、$192^3$ 网格下，典型结果为：

```text
Analytic overlap S(R):              0.7529427409
Numerical overlap:                  0.7529422641
Absolute overlap error:             4.77e-07
Bonding orbital norm:               0.9999988
Antibonding orbital norm:           0.9999955
Bonding-antibonding overlap error:  1.10e-08
```

---

## 等值面策略

项目支持两种阈值模式。

### 累计概率模式

自动寻找概率密度阈值 $\rho_c$，使高密度区域包含指定比例的总概率：

$$
\frac{
\sum_{\rho_{ijk}\ge\rho_c}
\rho_{ijk}\Delta V
}{
\sum_{ijk}
\rho_{ijk}\Delta V
}
\approx p.
$$

随后使用

$$
c=\sqrt{\rho_c}
$$

分别提取：

$$
\psi(\mathbf r)=+c,
\qquad
\psi(\mathbf r)=-c.
$$

默认设置为包围约 $90\%$ 的概率。

### 相对振幅模式

使用最大振幅的固定比例：

$$
c
=
\eta
\max|\psi|.
$$

其中 $\eta$ 由 `relative_amplitude` 参数控制。

---

## 可视化约定

- 红色：波函数正相位；
- 蓝色：波函数负相位；
- 白色球：氢原子核；
- 灰色线：核间轴。

颜色表示的是波函数相位，而不是正负概率。概率密度始终满足：

$$
|\psi|^2\ge0.
$$

---

## Notebook 结构

建议按以下单元格顺序运行：

```text
Cell 1  环境、依赖与版本信息
Cell 2  实验参数
Cell 3  物理模型
Cell 4  数值工具与验证
Cell 5  概率阈值算法
Cell 6  Marching Cubes 与网格导出
Cell 7  Plotly 三维可视化
Cell 8  完整计算流水线
Cell 9  执行实验
Cell 10 实验摘要
```

参数与计算逻辑相互分离。通常只需修改 `Cell 2`，然后重新运行实验单元格。

---

## 默认参数

```python
grid_config = GridConfig(
    half_length=8.0,
    resolution=192,
)

molecule_config = MoleculeConfig(
    internuclear_distance=1.4,
)

surface_config = SurfaceConfig(
    threshold_mode="probability",
    enclosed_probability=0.90,
    relative_amplitude=0.12,
    opacity=0.78,
    display_limit=4.5,
)
```

所有长度均使用原子单位，玻尔半径满足：

$$
a_0=1.
$$

---

## 安装依赖

在 Google Colab 中运行：

```bash
pip install jax scikit-image plotly trimesh
```

使用 NVIDIA GPU 时，应根据当前 CUDA 环境安装对应版本的 JAX。

主要依赖：

- JAX
- NumPy
- scikit-image
- Plotly
- Trimesh

---

## 快速运行

完成所有函数单元格后，执行：

```python
experiment_results = run_experiment(
    grid_config=grid_config,
    molecule_config=molecule_config,
    surface_config=surface_config,
    output_config=output_config,
)
```

程序会依次：

1. 构造三维规则网格；
2. 计算两个氢原子的 $1s$ 轨道；
3. 构造成键与反键分子轨道；
4. 验证归一化、重叠积分与正交性；
5. 求解等值面阈值；
6. 提取正负相位三角网格；
7. 输出交互式三维模型；
8. 保存实验元数据。

---

## 输出文件

默认输出目录：

```text
output_h2_reproducible/
├── experiment_metadata.json
├── h2_bonding_wavefunction.npy
├── h2_antibonding_wavefunction.npy
│
├── h2_bonding_orbital_positive.obj
├── h2_bonding_orbital_positive.ply
├── h2_bonding_orbital_combined.glb
├── h2_bonding_orbital.html
│
├── h2_antibonding_orbital_positive.obj
├── h2_antibonding_orbital_positive.ply
├── h2_antibonding_orbital_negative.obj
├── h2_antibonding_orbital_negative.ply
├── h2_antibonding_orbital_combined.glb
└── h2_antibonding_orbital.html
```

文件用途：

| 格式 | 用途 |
|---|---|
| `.npy` | 保存原始三维波函数 |
| `.html` | Plotly 交互式查看 |
| `.obj` | 通用三角网格 |
| `.ply` | 支持顶点属性与颜色 |
| `.glb` | 单文件三维场景，可导入 Blender 或 Three.js |
| `.json` | 保存参数、版本与数值验证结果 |

---

## 可复现性

每次实验都会生成 `experiment_metadata.json`，记录：

- 网格尺寸与空间范围；
- 核间距离；
- 等值面策略与阈值；
- Python 与依赖版本；
- JAX 计算设备；
- 轨道归一化结果；
- 重叠积分误差；
- 正交性误差；
- 网格顶点数、三角面数、面积与体积。

因此，每个导出的三维模型都可以追溯到其完整计算配置。

---

## 项目边界

当前版本采用最小基组的 LCAO 模型，仅使用两个氢原子 $1s$ 轨道。它适合展示：

- 分子轨道理论的基本结构；
- 成键与反键干涉；
- 节点面与波函数相位；
- 连续物理场的数值离散化；
- 三维标量场到三角网格的转换。

它不是高精度量子化学软件，也没有包含：

- 电子相关效应；
- 多基函数展开；
- Hartree–Fock 自洽场求解；
- Gaussian 基组；
- 核间排斥与完整势能曲线；
- 几何优化。

---

## 后续计划

- [ ] 扫描核间距离并生成轨道演化动画；
- [ ] 绘制成键与反键能级随 $R$ 的变化；
- [ ] 加入氢分子总电子密度；
- [ ] 支持 $2s$、$2p$ 与更多原子轨道；
- [ ] 支持 Gaussian Type Orbitals；
- [ ] 加入体渲染与最大密度投影；
- [ ] 生成静态高分辨率渲染图；
- [ ] 将计算核心封装为 Python 包；
- [ ] 添加自动化数值测试；
- [ ] 支持 Blender 与 WebGL 展示。

