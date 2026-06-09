# RKPM 椭球绕流：网格错位根因分析与修复

本文档记录椭球绕流（参考 Wang et al. 2025 JFM, Table 1/3）三项严重问题的根因定位、
修复方案、以及单层 RKPM 生成器与多层 AMR 求解器之间的对应性检验。

## 1. 问题现象

7 个攻角（0/25/35/45/55/65/90°）的系数曲线（`coeff.png`）相对参考文献有：

1. 展向力矩 **CT 峰值不在 45°**，且 CT 廓线不关于 45° 对称；
2. 攻角 **0° 与 90° 时 CT ≠ 0**（对称性要求应为 0）；
3. **阻力系数 CD 整体偏低**。

并且：**Reynolds 数越低，问题越严重**。

## 2. 根本原因：RKPM 生成器网格与求解器网格存在固定"原点错位"

`Tools/RKPM_weight` 的 RKPM 权重满足全部矩条件（0 阶矩 =1、1 阶矩 =0、2 阶矩 =0），
**但这些条件只在"生成器自己的局部网格"里成立**。生成器局部网格原点
`(sx, sy, sz) = (1.9, 2.15, 2.15)` 不是 `dx` 的整数倍：

```
sx/dx = 1.9 / 0.00292969 = 648.533   (非整数)
sy/dx = sz/dx = 2.15 / 0.00292969 = 733.866
```

`src/mapping.py` 用 `i_offset = int(sx*nxc/Lx) = int(sx/dx)` 把局部索引映射成全局索引时，
`int()` **截断**丢掉了小数部分（0.533, 0.867, 0.867 个网格），相当于把"原点在 1.9 的局部网格"
硬贴到"原点在 648·dx = 1.89844 的全局网格"上。于是在**求解器真实网格**里，1 阶矩不再为零。

### 实测证据（对现有 `rkpm_mappings.{id,lag}`，2830 个 marker）

| 量 | 应有值 | 实测 |
|---|---|---|
| `max\|Σw − 1\|`（0 阶矩，守恒力） | 0 | `1.3e-12` ✓ |
| `Σw·(x_euler − x_lag)/dx`（1 阶矩，守恒力矩） | 0 | **(−0.533, −0.867, −0.867) 个网格** ✗ |
| 上一项跨 marker 标准差 | — | `~1e-13`（即**所有 marker 完全相同的常数偏移**） |

实测常数偏移与预测 `(sx−offset·dx, …) = (0.00156, 0.00254, 0.00254) m` 逐位吻合。

## 3. 为什么这一个 bug 解释了全部三个现象

把"1 阶矩恒为常数 `shift`"翻译成物理：**每个 marker 的力实际作用在 `x_l + shift` 处**
（弥散核加权质心偏移 shift；插值同理在 `x_l + shift` 处采样速度）。对整个颗粒：

- **① 0°/90° CT ≠ 0**：寄生力矩 `T_spurious = shift × F_total`。0° 时 `F_total ≈ (F_drag,0,0)`，
  故 `T_z = −shift_y·F_drag ≠ 0`，且 `shift_y = 0.87` 网格、`F_drag` 又是最大的力。
- **② CT 不关于 45° 对称、峰值偏移**：`shift` 固定而 `F_total` 随攻角旋转，
  `shift × F_total` 随角度变化叠加到真实的 `∝ sin2θ` 廓线上，把对称钟形压歪。
- **③ CD 偏低**：两个最大偏移在 **y、z（垂直来流方向）**，相当于把有效无滑移面在横流方向
  内缩 ~0.87·dx，有效迎流尺寸变小 → 阻力系统性偏低。
- **④ 低 Re 更糟**：低 Re 时 CD 极大（~80），`shift×F_drag` 寄生力矩也极大，
  真实力矩很小被淹没；高 Re 时相对污染小。与"低 Re 永远不成功"一致。

## 4. 为什么之前的尝试都没用

1. **中心偏移 0.001（2.25→2.251）**：方向错。`2.25 = 768·dx` 本就落在求解器 cell face 上，
   cell-centered 网格关于 cell face 镜像对称，对 CT=0 **有利**；偏移反而破坏对称性。
2. **1/8 对称分布 marker**：marker 对称无法抵消一个**与 marker 无关的固定网格偏移**。
3. **改用 RKPM 索引（index[13]）**：`.lag` 里的索引本身就编码了那套错位网格，没碰到病根。
4. **力的归一化**：与前三个形状问题无关，可后处理。

## 5. 修复方案（已实现）

### (a) 生成器网格对齐 —— 根因修复
- `Tools/RKPM_weight/main.py`：把箱体原点吸附到 `dx` 整数倍
  `sx = round(sx/dx)*dx`（y/z 同），并打印对齐后网格数（必须为整数）。
- `Tools/RKPM_weight/src/mapping.py`：全局索引偏移 `int(...)` 改为 `round(...)`，
  原点对齐后精确无误差。

### (b) C++ 核改用每条目自带索引 —— `Source/DiffusedIB_Parallel.cpp`
- `VelocityInterpolationRKPM_cir` / `ForceSpreadingRKPM_cir`：不再假设"27 点立方体、
  中心在 index[13]"，改为对每个 stencil 条目用它自己的 `rkpm.index[0/1/2]` 访问欧拉单元；
  插值与弥散严格互为转置。
- `ResolveWithRPKM` 扁平化加安全填充（空槽 weight=0 但带合法索引），固定长度循环不会越界。

### (c) 验收脚本 —— `Tools/RKPM_weight/verify_rkpm_alignment.py`
- 在**求解器坐标系**下检验 0 阶矩（=1）与 1 阶矩（=0），并自动诊断"常数偏移 = 网格错位"特征。

### 附带（正确性必需）—— `inputs.3d.flow_past_ellipsoid`
- `particle_inputs.y/z` 由 `2.251` 改回 `2.25`（落在 cell face，保 y/z 镜像对称 → 0°/90° CT=0）。

## 6. 单层生成器 ↔ 多层 AMR 求解器 对应性检验

IAMReX 是多层自适应网格，IB 在第 `amr.max_level` 层（最细层）的**单层 MultiFab `S_new`** 上进行
（`NavierStokes.cpp:2687` 处 `level == ParticleFinestLevel()`，容器以该层 `geom/grids` 为其第 0 层）。
生成器是单层均匀网格。三项对应性检验结果：

| 检验项 | 要求 | 结果 |
|---|---|---|
| (A) dx 对应 | 生成器 `Lx/nxc` = 求解器最细层 `(prob_hi−prob_lo)/(n_cell·2^max_level)` | `0.00292969 == 0.00292969` ✓，反推 IB 层 = max_level = 3 ✓ |
| (B) stencil 覆盖 | 所有 27 点全落在被细化到最细层的区域内且远离粗细边界 | 全部在 tracer 细化盒内，最小余量 **44 个网格** ✓ |
| (C) 索引↔坐标 | `.lag` 全局索引落在求解器最细层为细化盒创建的索引窗口内 | ✓ |

**对应性成立**，但依赖一个代码里没有运行时检查的**隐式不变量**：

```
Lx/nxc  ==  (prob_hi − prob_lo) / (n_cell · 2^max_level)
```

会让对应性悄悄失效的改动（换 Re/换算例时尤其当心）：
- 改了 `n_cell` / `max_level` / 域尺寸却没同步改生成器 `Lx,nxc` → dx 不再相等，索引全错位且无报错；
- 细化判据从 `in_box` 改成涡量/阈值 → 细化区可能出现内部空洞或覆盖不全 → 读到粗细插值 ghost；
- 细化盒缩到贴近粒子 → marker stencil 触到粗细边界，精度下降。

建议保险：生成器侧把 dx 等式做成断言；求解器侧在首次 `InteractWithEuler` 时断言所有 stencil 索引
落在 `S_new` 的 boxArray（含 ghost）内，越界即 `amrex::Abort`。

## 7. 验证结果

| 阶段 | `max\|Σw(x−x_l)\|/dx`（1 阶矩，网格） | `verify_rkpm_alignment.py` |
|---|---|---|
| 修复前（现有数据） | (0.533, 0.867, 0.867) | **FAIL**（诊断出常数偏移） |
| 对齐后重生成 | ~1e-13 | **PASS** |

- 用对齐后的生成器对现有 2830 点重算权重：求解器坐标系 1 阶矩 0.87 网格 → **1e-13**；
  生成器自检 Force/Torque Conservation Error = 3.6e-15 / 4.3e-14。
- `Source/DiffusedIB_Parallel.cpp` 改动已在 `Tutorials/FlowPastSphere` 编译链接通过（`SUCCESS`）。

## 8. 涉及文件

| 文件 | 改动 | 对应修复 |
|---|---|---|
| `Source/DiffusedIB_Parallel.cpp` | 两个 RKPM 核改用每条目索引；扁平化安全填充 | (b) |
| `Tools/RKPM_weight/main.py` | 箱体原点吸附到 dx 整数倍 | (a) |
| `Tools/RKPM_weight/src/mapping.py` | 全局索引偏移 `int`→`round` | (a) |
| `Tools/RKPM_weight/verify_rkpm_alignment.py` | 新增：守恒性/对齐验收脚本 | (c) |
| `Tutorials/Jang_E2830Re03p65/inputs.3d.flow_past_ellipsoid` | 粒子中心 y/z 2.251→2.25 | 附带 |
| `Tutorials/Jang_E2830Re03p65/RKPM_grid_alignment_fix.md` | 新增：本说明 | — |

## 9. 后续步骤

1. 对每个攻角的椭球点云，用修好的生成器重生成 `rkpm_mappings.{id,lag}`
   （务必保证 `Lx/nxc == (prob_hi−prob_lo)/(n_cell·2^max_level)`）。
2. 每个角度先跑 `python verify_rkpm_alignment.py` 必须 **PASS** 再投入仿真。
3. 重新编译并重跑，先验证 **0°/90° 的 CT≈0**，再补齐全部角度对比 Wang et al. (2025)。
4. 若 CD 仍偏低：对齐修复应回收大部分；剩余再试 `LOOP_NS=6~8` 或加密 marker 至间距 ~0.8–1.0·dx。
