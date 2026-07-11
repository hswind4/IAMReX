import numpy as np
from src import visual
import sys
import ast

# 读取拉格朗日点云文件 (xp,yp,zp)
# 支持空白分隔、以 # 开头注释行的格式（如 test_cylinder_centers.txt）：
#   # X Y Z
#   -0.0087703851 0.0371783379 0.0369941932
#   ...
def read_geometry_file(file_path):
    points = np.atleast_2d(np.loadtxt(file_path, comments='#'))
    if points.size == 0:
        raise ValueError(f"几何文件为空或无法解析: {file_path}")
    if points.shape[1] != 3:
        raise ValueError(
            f"几何文件需包含3列(X Y Z)，实际 {points.shape[1]} 列: {file_path}")
    return points

# 读取id文件拉格朗日点位置(xp,yp,zp)
# def load_lagrangian_from_id_file(filename):
#     """
#     读取 .id 文件并转换为 (N, 3) 的 numpy 数组
#     """
#     # 1. 读取整个文件内容
#     with open(filename, 'r') as f:
#         content = f.read()
#
#     # 2. 将字符串转换为 Python 字典
#     # 你的文件格式是 valid 的 Python 字典语法: { 0: (x,y,z), ... }
#     try:
#         data_dict = ast.literal_eval(content)
#     except Exception as e:
#         print(f"解析文件失败: {e}")
#         return None
#
#     # 3. 确定数组大小 N
#     # 假设 ID 是从 0 到 N-1 连续的，或者取最大 ID + 1
#     if not data_dict:
#         return np.empty((0, 3))
#
#     num_points = len(data_dict)
#
#     # 4. 初始化 numpy 数组
#     lagrangian_points = np.zeros((num_points, 3))
#
#     # 5. 填充数据
#     # 直接利用字典的 key 作为数组的 index
#     for lag_id, coords in data_dict.items():
#         lagrangian_points[lag_id] = coords
#
#     return lagrangian_points

# 查找点在网格坐标中的索引
# 输入：
#   points: 需要查找的点坐标（一维或二维数组）
#   coords: 网格坐标数组
# 输出：
#   idxs: 每个点在coords中的区间索引
def find_grid_indices(points, coords):
    idxs = np.searchsorted(coords, points, side='right') - 1
    if np.any(idxs < 0) or np.any(idxs >= len(coords) - 1):
        raise ValueError("Some points are out of bounds.")
    return idxs

# 计算椭球表面积Knud Thomsen 近似公式
def ellipsoid_area_approx(a, b, c):
    """
    使用 Knud Thomsen 公式计算近似表面积。
    """
    p = 1.6075
    term = ((a * b)**p + (a * c)**p + (b * c)**p) / 3.0
    return 4 * np.pi * (term**(1 / p))

def cylinder_area(radius, height):
    """
    计算圆柱的总表面积（包括上下两个底面）。

    参数:
        radius (float): 圆柱底面半径
        height (float): 圆柱高度

    返回:
        float: 总面积 = 2 * pi * radius * (radius + height)
    """
    return 2 * np.pi * radius * (radius + height)

# 生成三维欧拉网格和拉格朗日点
#
# 设计：直接在求解器最细网格的全局坐标系下工作，不再选取局部子区域、也不再施加
# 局部->全局的索引偏置（旧的 int(sx/dx) 截断会引入歧义）。
#   1. 从 --inputs 读 prob_lo/prob_hi/n_cell/max_level，最细网格尺寸
#      dx_finest = (prob_hi - prob_lo) / (n_cell * 2**max_level)（每层网格独立编号，
#      第 level 层 dx = (prob_hi-prob_lo)/(n_cell*2**level)，索引 = floor((x-prob_lo)/dx)）。
#   2. 拉格朗日点云（世界坐标）的 min/max 给出极限包裹区域；在"最细网格索引"下
#      向外扩 2 个网格（min-2, max+2）作为欧拉计算区域，并夹断到求解域范围。
#   3. 欧拉网格用全局单元中心 (idx+0.5)*dx+prob_lo 生成，索引天然是全局的，
#      无需任何偏置。RKPM 权重计算（window.py）只依赖相对坐标，不受原点影响。
#
# 输入参数：
#   prob_lo: 求解域下界 (3,)
#   prob_hi: 求解域上界 (3,)
#   dx_finest: 最细网格单元尺寸 (3,) = (prob_hi-prob_lo)/(n_cell*2**max_level)
#   geometry_file: 拉格朗日点云文件路径。为 None 时从 rkpm_mappings.id 读取
#   center: (3,) 几何体在世界坐标的中心。非 None 时认为点云是 body frame（近原点），
#           先整体平移到 center，再绕 center 旋转。为 None 时点云按世界坐标使用，绕世界原点旋转。
#   angle: 绕 z 轴的旋转角度（度，默认 0，仅改变 x,y）。平移之后再旋转；body-frame 枢轴
#          为 center，否则为世界原点 (0,0,0)。
# 输出：
#   eulerian_points: 所有欧拉网格点坐标 (N, 3)
#   lagrangian_points: 所有拉格朗日点坐标 (Ne, 3)
#   nearest_grid_points: 每个拉格朗日点最近的欧拉点坐标 (Ne, 3)
#   delta_I, eta_I, theta_I: 每个拉格朗日点的支持域参数（一维数组）
#   all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
def generate_grid(prob_lo, prob_hi, dx_finest, geometry_file=None, center=None, angle=0.0):

    prob_lo = np.asarray(prob_lo, dtype=float)
    prob_hi = np.asarray(prob_hi, dtype=float)
    dx_finest = np.asarray(dx_finest, dtype=float)
    dx, dy, dz = dx_finest
    # 每个方向最细网格的单元总数（用于夹断到求解域范围）
    n_fine = np.round((prob_hi - prob_lo) / dx_finest).astype(int)

    # --- 拉格朗日点云 ---
    if geometry_file is not None:
        lagrangian_points = read_geometry_file(geometry_file)
    else:
        raise NameError("要配合拉格朗日点坐标文件来使用！")
        # lagrangian_points = load_lagrangian_from_id_file('rkpm_mappings.id')
    # body frame -> 世界坐标：先整体平移到 center
    if center is not None:
        center = np.asarray(center, dtype=float)
        lagrangian_points = lagrangian_points + center
    # 绕 z 轴旋转（仅改变 x,y）：平移之后再旋转。body-frame 枢轴为 center，否则为世界原点
    if angle:
        pivot = center if center is not None else np.zeros(3)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[ c, -s, 0.],
                       [ s,  c, 0.],
                       [0., 0., 1.]])
        lagrangian_points = (lagrangian_points - pivot) @ Rz.T + pivot
    visual.PointCloud(lagrangian_points)

    # --- 包裹区域：点云 min/max，最细网格索引向外扩 2 个网格 ---
    pmin = lagrangian_points.min(axis=0)
    pmax = lagrangian_points.max(axis=0)
    # 点云须落在求解域 [prob_lo, prob_hi] 内，否则其全局单元索引会越界
    if np.any(pmin < prob_lo) or np.any(pmax > prob_hi):
        raise ValueError(
            f"点云越出求解域 [prob_lo, prob_hi]：\n"
            f"  prob_lo = {prob_lo.tolist()}\n  prob_hi = {prob_hi.tolist()}\n"
            f"  点云 min = {pmin.tolist()}\n  点云 max = {pmax.tolist()}\n"
            f"请检查 --geometry 坐标系，或用 --body-frame 将点云平移到域内。"
        )
    ranges = pmax - pmin  # = np.ptp(lagrangian_points, axis=0)
    print('ranges', ranges[0], ranges[1], ranges[2])
    print('ranges', ranges[0]/2, ranges[1]/2, ranges[2]/2)

    # 点云在最细网格下的全局单元索引，向外扩 2 个网格并夹断到 [0, n_fine-1]
    i_lo = int(np.floor((pmin[0] - prob_lo[0]) / dx)) - 2
    j_lo = int(np.floor((pmin[1] - prob_lo[1]) / dy)) - 2
    k_lo = int(np.floor((pmin[2] - prob_lo[2]) / dz)) - 2
    i_hi = int(np.floor((pmax[0] - prob_lo[0]) / dx)) + 2
    j_hi = int(np.floor((pmax[1] - prob_lo[1]) / dy)) + 2
    k_hi = int(np.floor((pmax[2] - prob_lo[2]) / dz)) + 2
    i_lo, j_lo, k_lo = max(0, i_lo), max(0, j_lo), max(0, k_lo)
    i_hi = min(int(n_fine[0]) - 1, i_hi)
    j_hi = min(int(n_fine[1]) - 1, j_hi)
    k_hi = min(int(n_fine[2]) - 1, k_hi)
    print(f"[grid] finest index range: i[{i_lo},{i_hi}] j[{j_lo},{j_hi}] k[{k_lo},{k_hi}] "
          f"-> cells ({i_hi-i_lo+1}, {j_hi-j_lo+1}, {k_hi-k_lo+1})")

    # --- 欧拉网格：全局单元中心 (idx+0.5)*dx+prob_lo ---
    # x/y/z 为单元边界（供 find_grid_indices 查找），xc/yc/zc 为单元中心
    x = prob_lo[0] + np.arange(i_lo, i_hi + 2) * dx
    y = prob_lo[1] + np.arange(j_lo, j_hi + 2) * dy
    z = prob_lo[2] + np.arange(k_lo, k_hi + 2) * dz
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])

    XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')

    # 计算网格长度与体积
    Delta_V = np.full((len(x)-1, len(y)-1, len(z)-1), dx * dy * dz)

    # 获取欧拉坐标
    eulerian_points = np.vstack([XC.ravel(), YC.ravel(), ZC.ravel()]).T

    # 计算拉格朗日点面积和厚度
    Ne = len(lagrangian_points)
    area = ellipsoid_area_approx(ranges[0]/2, ranges[1]/2, ranges[2]/2) / Ne
    # area = ellipsoid_area_approx(0.0437, 0.0437, 0.0655) / Ne #椭球表面积公式
    # area = cylinder_area(0.03815, 0.1145) / Ne #圆柱表面积公式
    thickness = min(dx, dy, dz)
    V_lag = area * thickness
    print(f"Vl: {V_lag}, area:{area}, frac: {V_lag / Delta_V[0][0][0]}")

    # 获取网格形状
    grid_shape = XC.shape

    # 搜索拉格朗日点最近的欧拉网格点（包含该点的全局单元 = floor((x-prob_lo)/dx)）
    indices_ijk = np.floor((lagrangian_points - prob_lo) / dx_finest).astype(int)
    # 转为本地数组下标（i_lo 等为精确整数，仅用于数组寻址，非浮点偏置）
    local_ijk = (indices_ijk[:, 0] - i_lo, indices_ijk[:, 1] - j_lo, indices_ijk[:, 2] - k_lo)
    nearest_indices = np.ravel_multi_index(local_ijk, dims=grid_shape)
    nearest_grid_points = eulerian_points[nearest_indices]

    # 计算delta_I, eta_I, theta_I
    delta_I = np.full(Ne, dx + (1 / 1000) * dx)
    eta_I = np.full(Ne, dy + (1 / 1000) * dy)
    theta_I = np.full(Ne, dz + (1 / 1000) * dz)

    # 计算 all_S_I
    all_S_I = []
    for idx in range(Ne):

        nearest_point = nearest_grid_points[idx]
        delta_I_lag = delta_I[idx]
        eta_I_lag = eta_I[idx]
        theta_I_lag = theta_I[idx]
        # 找到位于矩形区域内的欧拉网格点
        mask_x = np.abs(eulerian_points[:, 0] - nearest_point[0]) < 1.5 * delta_I_lag
        mask_y = np.abs(eulerian_points[:, 1] - nearest_point[1]) < 1.5 * eta_I_lag
        mask_z = np.abs(eulerian_points[:, 2] - nearest_point[2]) < 1.5 * theta_I_lag

        S_I_points = eulerian_points[mask_x & mask_y & mask_z]

        # 向量化查找每个点的网格索引
        i = find_grid_indices(S_I_points[:, 0], x)
        j = find_grid_indices(S_I_points[:, 1], y)
        k = find_grid_indices(S_I_points[:, 2], z)

        volume_points = Delta_V[i,j,k]

        # 合并为 (x, y, z, volume)
        S_I = np.column_stack((S_I_points, volume_points.reshape(-1, 1)))
        all_S_I.append(S_I)

    return eulerian_points, Ne, lagrangian_points, nearest_grid_points, delta_I, eta_I, theta_I, all_S_I, V_lag
