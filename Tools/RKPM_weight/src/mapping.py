"""
RKPM 3D 映射构建模块

用于构建三组关键映射：
1. 拉格朗日点ID到坐标映射
2. 拉格朗日点到欧拉网格映射 (force spreading)
3. 欧拉网格到拉格朗日点映射 (velocity interpolation)

欧拉点本身就是求解器最细网格的全局单元中心，因此全局单元索引直接由坐标算出
i = floor((x - prob_lo) / dx_finest)，不再使用局部子区域的索引偏置
（旧的 int(sx/dx) 截断会在原点非整数倍 dx 时引入歧义）。
"""

import numpy as np
from typing import Dict, List, Union


def build_lagrangian_id_to_coord_map(lagrangian_points: np.ndarray) -> Dict[int, tuple]:
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
    prob_lo: np.ndarray,
    dx_finest: np.ndarray,
    V_lag: float
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """
    构建拉格朗日点到欧拉网格的映射 (Force Spreading用)

    全局单元索引直接由坐标算出 i = floor((x - prob_lo) / dx_finest)。
    all_S_I 中的欧拉点即最细网格的全局单元中心，故 floor 恰好还原其全局单元号，
    无需局部->全局的偏置。

    输入参数：
        lagrangian_points: (Ne, 3) 所有拉格朗日点坐标（世界坐标）
        all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
        all_modified_w: 所有拉格朗日点的修正窗口函数值列表
        prob_lo: 求解域下界 (3,)
        dx_finest: 最细网格单元尺寸 (3,)
        V_lag: 单个拉格朗日点的体积

    输出：
        lag_to_eul_map: {lag_id: [{"i": int, "j": int, "k": int, "w": float, "Vcell": float, "eps": float}, ...]}
    """
    prob_lo = np.asarray(prob_lo, dtype=float)
    dx_finest = np.asarray(dx_finest, dtype=float)
    lag_to_eul_map = {}

    for lag_id in range(len(lagrangian_points)):
        S_I = all_S_I[lag_id]  # 支持域内的欧拉点及体积信息
        modified_w = all_modified_w[lag_id]  # 修正窗口函数值

        eulerian_data = []

        for m, (x_mn, y_mn, z_mn, Vcell) in enumerate(S_I):
            # 全局单元索引 = floor((单元中心 - prob_lo) / dx)
            i = int(np.floor((x_mn - prob_lo[0]) / dx_finest[0]))
            j = int(np.floor((y_mn - prob_lo[1]) / dx_finest[1]))
            k = int(np.floor((z_mn - prob_lo[2]) / dx_finest[2]))

            # 获取权重
            w = modified_w[m]

            # 添加到映射中
            eulerian_data.append({
                "i": i,
                "j": j,
                "k": k,
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
    id_to_coord_map: Dict[int, tuple],
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
