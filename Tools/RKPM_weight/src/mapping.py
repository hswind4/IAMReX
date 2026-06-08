"""
RKPM 3D 映射构建模块

用于构建三组关键映射：
1. 拉格朗日点ID到坐标映射
2. 拉格朗日点到欧拉网格映射 (force spreading)
3. 欧拉网格到拉格朗日点映射 (velocity interpolation)

针对高性能计算环境优化，支持HDF5、NPZ和内存映射格式。
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import pickle
from collections import defaultdict


def extract_grid_coordinates(eulerian_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从欧拉网格点坐标中提取x、y、z坐标数组
    
    输入参数：
        eulerian_points: (N, 3) 所有欧拉网格点坐标
    
    输出：
        x_coords: 唯一的x坐标值，已排序
        y_coords: 唯一的y坐标值，已排序  
        z_coords: 唯一的z坐标值，已排序
    """
    x_coords = np.unique(eulerian_points[:, 0])
    y_coords = np.unique(eulerian_points[:, 1])
    z_coords = np.unique(eulerian_points[:, 2])
    
    # 确保坐标数组已排序
    x_coords = np.sort(x_coords)
    y_coords = np.sort(y_coords)
    z_coords = np.sort(z_coords)
    
    return x_coords, y_coords, z_coords


def get_grid_indices(point: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray) -> Tuple[int, int, int]:
    """
    将欧拉点坐标转换为网格索引(i,j,k)
    
    输入参数：
        point: (3,) 欧拉点坐标 [x, y, z]
        x_coords: x坐标数组
        y_coords: y坐标数组
        z_coords: z坐标数组
    
    输出：
        (i, j, k): 网格索引元组
    """
    x, y, z = point
    
    # 使用searchsorted找到最近的网格索引
    i = np.searchsorted(x_coords, x, side='right') - 1
    j = np.searchsorted(y_coords, y, side='right') - 1
    k = np.searchsorted(z_coords, z, side='right') - 1
    
    # 确保索引在有效范围内
    i = max(0, min(i, len(x_coords) - 2))
    j = max(0, min(j, len(y_coords) - 2))
    k = max(0, min(k, len(z_coords) - 2))
    
    return int(i), int(j), int(k)

def build_lagrangian_id_to_coord_map(lagrangian_points: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
    """
    构建拉格朗日点ID到坐标的映射
    
    输入参数：
        lagrangian_points: (Ne, 3) 所有拉格朗日点坐标
    
    输出：
        id_to_coord_map: {id: (xp, yp, zp)}
    """
    id_to_coord_map = {}
    
    for lag_id, coord in enumerate(lagrangian_points):
        id_to_coord_map[lag_id] = tuple(coord)
    
    return id_to_coord_map

def build_lag_to_eul_map(
    lagrangian_points: np.ndarray,
    all_S_I: List[np.ndarray],
    all_modified_w: List[List[float]],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    sx: float, sy: float, sz: float,
    Lx: float, Ly: float, Lz: float,
    nxc: int, nyc: int, nzc: int,
    V_lag: float
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """
    构建拉格朗日点到欧拉网格的映射 (Force Spreading用)

    输入参数：
        lagrangian_points: (Ne, 3) 所有拉格朗日点坐标
        all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
        all_modified_w: 所有拉格朗日点的修正窗口函数值列表
        epsilon: (Ne) ε修正因子数组
        x_coords, y_coords, z_coords: 网格坐标数组

    输出：
        lag_to_eul_map: {lag_id: [{"i": int, "j": int, "k": int, "w": float, "Vcell": float, "eps": float}, ...]}
    """
    lag_to_eul_map = {}

    i_offset = int(sx * nxc / Lx)
    j_offset = int(sy * nyc / Ly)
    k_offset = int(sz * nzc / Lz)    

    for lag_id in range(len(lagrangian_points)):
        S_I = all_S_I[lag_id]  # 支持域内的欧拉点及体积信息
        modified_w = all_modified_w[lag_id]  # 修正窗口函数值

        eulerian_data = []

        for m, (x_mn, y_mn, z_mn, Vcell) in enumerate(S_I):
            # 获取网格索引
            point = np.array([x_mn, y_mn, z_mn])
            i, j, k = get_grid_indices(point, x_coords, y_coords, z_coords)

            # 获取权重
            w = modified_w[m]

            # 添加到映射中
            eulerian_data.append({
                "i": i + i_offset,
                "j": j + j_offset,
                "k": k + k_offset,
                "w": float(w),
                "Vcell": 1.0,
                "eps": float(V_lag) / float(Vcell)
            })

        # 在这里排序：按 k → j → i 顺序 (k主序)
        sort_keys = [(item["k"], item["j"], item["i"]) for item in eulerian_data]
        # 转为 numpy 数组便于 lexsort
        sort_keys = np.array(sort_keys)  # shape: (N, 3)

        # 使用 lexsort：优先级 k > j > i
        # 注意：lexsort 从最后一列开始排序，所以传入 (i, j, k)
        indices = np.lexsort((sort_keys[:, 0], sort_keys[:, 1], sort_keys[:, 2]))

        # 按排序后的索引重新组织列表
        eulerian_data = [eulerian_data[idx] for idx in indices]

        # 存入 map
        lag_to_eul_map[lag_id] = eulerian_data

    return lag_to_eul_map

def save_mappings_txt(
    id_to_coord_map: Dict[int, Tuple[float, float, float]],
    lag_to_eul_map: Dict[int, List[Dict[str, Union[int, float]]]],
    filename: str
) -> None:
    """
    保存映射到txt文件

    输入参数：
        id_to_coord_map: 拉格朗日ID到坐标映射
        lag_to_eul_map: 拉格朗日到欧拉映射
        filename: 输出文件名
    """
    # 1. 保存拉格朗日点ID的坐标映射（id_to_coord_map）
    with open(filename + ".id", 'w') as f:
        f.write("{\n")
        for ids, (x, y, z) in id_to_coord_map.items():
            f.write(f"    {ids}: ({x}, {y}, {z}),\n")
        f.write("}")

    # 2. 保存 lag_to_eul_map 为指定格式的 .txt 文件
    with open(filename + ".lag", 'w') as f:
        f.write("{\n")
        for lag_id, eul_list in lag_to_eul_map.items():
            # 写入 lag_id 和对应的欧拉点列表
            f.write(f"    {lag_id}: [\n")
            for eul_info in eul_list:
                # 格式化每个欧拉点的字典数据
                line = (
                    f"        {{"
                    f"\"i\": {eul_info['i']}, "
                    f"\"j\": {eul_info['j']}, "
                    f"\"k\": {eul_info['k']}, "
                    f"\"w\": {eul_info['w']}, "
                    f"\"Vcell\": {eul_info['Vcell']}, "
                    f"\"eps\": {eul_info['eps']}"
                    f"}},\n"
                )
                f.write(line)
            f.write("    ],\n")  # 结束当前 lag_id 的列表
        f.write("}")  # 结束整个字典
