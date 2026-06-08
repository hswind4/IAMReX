import numpy as np
from src import visual
import sys
import pandas as pd
import ast

# 读取csv文件拉格朗日点位置(xp,yp,zp)
def read_point_cloud(file_name):
    file_path = file_name  # 替换为你的CSV文件路径
    # 根据CSV文件的实际分隔符调整sep参数（常见的有','，'\t'等）
    df = pd.read_csv(file_path, sep=',', skiprows=7, header=None, encoding='gbk')

    # 根据实际列名提取XYZ数据 - 修改这里的列名以匹配你的CSV文件
    # 如果列名中有空格或特殊字符，请使用精确列名
    xp = df.iloc[:, 2].copy()
    yp = df.iloc[:, 3].copy()
    zp = df.iloc[:, 4].copy()

    lagrangian_points = np.vstack([xp,yp,zp]).T

    return lagrangian_points

# 读取id文件拉格朗日点位置(xp,yp,zp)
def load_lagrangian_from_id_file(filename):
    """
    读取 .id 文件并转换为 (N, 3) 的 numpy 数组
    """
    # 1. 读取整个文件内容
    with open(filename, 'r') as f:
        content = f.read()
    
    # 2. 将字符串转换为 Python 字典
    # 你的文件格式是 valid 的 Python 字典语法: { 0: (x,y,z), ... }
    try:
        data_dict = ast.literal_eval(content)
    except Exception as e:
        print(f"解析文件失败: {e}")
        return None

    # 3. 确定数组大小 N
    # 假设 ID 是从 0 到 N-1 连续的，或者取最大 ID + 1
    if not data_dict:
        return np.empty((0, 3))
        
    num_points = len(data_dict)
    # 或者为了保险起见，使用 max(data_dict.keys()) + 1
    
    # 4. 初始化 numpy 数组
    lagrangian_points = np.zeros((num_points, 3))
    
    # 5. 填充数据
    # 直接利用字典的 key 作为数组的 index
    for lag_id, coords in data_dict.items():
        lagrangian_points[lag_id] = coords
        
    return lagrangian_points

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
# 输入参数：
#   sx, sy, sz: 局部区域起始坐标
#   Lx, Ly, Lz: 局部区域长度
#   nxc, nyc, nzc: 局部区域网格数
#   center: 中心坐标
#   readpc: 读取文件类型
# 输出：
#   eulerian_points: 所有欧拉网格点坐标 (N, 3)
#   lagrangian_points: 所有拉格朗日点坐标 (Ne, 3)
#   nearest_grid_points: 每个拉格朗日点最近的欧拉点坐标 (Ne, 3)
#   delta_I, eta_I, theta_I: 每个拉格朗日点的支持域参数（一维数组）
#   all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
def generate_grid(sx, sy, sz, Lx, Ly, Lz, nxc, nyc, nzc, center, readpc=0):

    dx = Lx/nxc
    dy = Ly/nyc
    dz = Lz/nzc

    # 生成欧拉网格
    x = np.arange(sx, sx + Lx + dx, dx)
    y = np.arange(sy, sy + Ly + dy, dy)
    z = np.arange(sz, sz + Lz + dz, dz)

    # 计算每个方向上的中心点
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])

    XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')

    # 计算网格长度与体积
    Delta_V = np.full((len(x)-1, len(y)-1, len(z)-1), dx * dy * dz)
    
    # 获取欧拉坐标
    eulerian_points = np.vstack([XC.ravel(), YC.ravel(), ZC.ravel()]).T

    if readpc==1:
        points = read_point_cloud('Cylinder1200.csv')
        #points = read_point_cloud('Ellipsoid1200.csv')
        ranges = np.ptp(points,axis=0)
        print('ranges',ranges[0],ranges[1],ranges[2])

        # rotate
        theta = np.radians(90)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,        1]
        ])

        points = np.dot(points, rotation_matrix.T)        
        lagrangian_points = np.array(points) + np.array(center)

        visual.PointCloud(lagrangian_points)
    else:
        lagrangian_points = load_lagrangian_from_id_file('rkpm_mappings.id')
        ranges = np.ptp(lagrangian_points,axis=0)
        visual.PointCloud(lagrangian_points)

    print('ranges',ranges[0]/2,ranges[1]/2,ranges[2]/2)
    # 计算拉格朗日点面积和厚度
    Ne = len(lagrangian_points)
    area = ellipsoid_area_approx(ranges[0]/2, ranges[1]/2, ranges[2]/2) / Ne
    # area = ellipsoid_area_approx(0.0437, 0.0437, 0.0655) / Ne #椭球表面积公式
    # area = cylinder_area(0.03815, 0.1145) / Ne #圆柱表面积公式
    thickness = min(dx,dy,dz)
    V_lag = area*thickness
    print(f"Vl: {V_lag}, area:{area}, frac: {V_lag / Delta_V[0][0][0]}")

    # 获取 X 和 Y 网格的形状
    grid_shape = XC.shape    

    # 搜索拉格朗日点最近的欧拉网格点
    plo = np.array([sx, sy, sz])
    d_grid = np.array([Lx/nxc, Ly/nyc, Lz/nzc])
    indices_ijk = np.floor((lagrangian_points - plo) / d_grid).astype(int)
    nearest_indices = np.ravel_multi_index((indices_ijk[:,0], indices_ijk[:,1], indices_ijk[:,2]), dims=grid_shape)
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
