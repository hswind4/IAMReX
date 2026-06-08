import numpy as np

# 测试函数
# 输入参数：
#   x, y: 浮点数或数组，表示坐标
# 输出：
#   返回 sin(pi*x)*cos(pi*y) 的值
def test_function(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# 计算无穷范数误差
# 输入参数：
#   original_values: 原始函数值数组
#   interpolated_values: 插值后函数值数组
# 输出：
#   返回两者的最大绝对误差
def compute_infinity_norm_error(original_values, interpolated_values):
    diff = original_values - interpolated_values
    return np.max(np.abs(diff))

def compute_error_volume(lagrangian_points, all_S_I, all_modified_w, V_lag):

    interpolated_volumn = np.zeros(len(lagrangian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            Delta_V = data[3]
            interpolated_volumn[idx] += Delta_V * modified_w[j] * V_lag / Delta_V

    relavtive_error = np.abs(interpolated_volumn - V_lag) / V_lag
    print(f"Relative volumn error: Min: {np.min(relavtive_error):.6e}, Max: {np.max(relavtive_error):.6e}")

    return relavtive_error
    
# 误差分析主函数
# 输入参数：
#   eulerian_points: 所有欧拉网格点坐标 (N, 3)
#   lagrangian_points: 所有拉格朗日点坐标 (Ne, 3)
#   all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
#   all_modified_w: 所有拉格朗日点的修正窗口函数值列表
#   epsilon: ε系数数组
# 输出：
#   返回无穷范数误差
def compute_error(eulerian_points, lagrangian_points, all_S_I, all_modified_w, V_lag):
    # original_values = test_function(lagrangian_points[:, 0], lagrangian_points[:, 1])
    original_values = np.ones(len(lagrangian_points))

    dispersion_values = np.zeros(len(eulerian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3]
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            dispersion_values[ide] += original_values[idx] * modified_w[j] * V_lag / Delta_V

    interpolated_values = np.zeros(len(lagrangian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3]
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            interpolated_values[idx] += dispersion_values[ide] * modified_w[j]

    # check force conservation
    force_lagrangian = np.sum(original_values * V_lag)
    volumns = np.full(len(eulerian_points), fill_value=all_S_I[0][0][3])
    force_euler = np.sum(dispersion_values * volumns)
    print('force check',force_lagrangian,force_euler, abs(force_lagrangian - force_euler)/force_lagrangian)

    return compute_infinity_norm_error(original_values, interpolated_values)

def compute_conservation_check(center, eulerian_points, lagrangian_points, all_S_I, all_modified_w, V_lag):
    """
    同时验证力（Force）和力矩（Moment/Torque）的守恒性
    """
    # 1. 预处理：建立网格坐标到索引的映射，加速查找 (优化原代码的O(N^2)查找)
    # 将坐标转换为tuple以便作为字典key，注意浮点数精度问题，但在同一流程中通常一致
    coord_map = {tuple(p): i for i, p in enumerate(eulerian_points)}
    
    # 2. 定义拉格朗日点上的测试力场 (Vector Force Field)
    # 使用“常量力 + 旋转项(omega x r)”构造非对称力场，避免总力矩接近0导致相对误差失真
    rel_pos = lagrangian_points - center
    omega = np.array([0.7, -0.3, 0.5])  # 旋转向量，控制非对称强度
    F_const = np.array([1.0, 0.8, 1.2]) # 常量偏置，避免总力过小
    F_lag = F_const + np.cross(np.tile(omega, (len(lagrangian_points), 1)), rel_pos)
    
    # 3. 计算拉格朗日侧的总力和总力矩
    # Force = sum(F_l * V_l)
    total_force_lag = np.sum(F_lag * V_lag, axis=0)
    
    # Torque = sum((X_l x F_l) * V_l)
    # np.cross(A, B) 计算叉乘
    torques_lag = np.cross(lagrangian_points - center, F_lag)
    total_torque_lag = np.sum(torques_lag * V_lag, axis=0)
    
    print(f"Lagrangian Total Force: {total_force_lag}")
    print(f"Lagrangian Total Torque: {total_torque_lag}")

    # 4. 将力弥散到欧拉网格 (Dispersion)
    # 初始化网格上的力密度 accumulator 和 网格体积记录器
    f_euler = np.zeros((len(eulerian_points), 3))
    grid_volumes = np.zeros(len(eulerian_points))
    
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        F_l = F_lag[idx] # 当前点的力向量
        
        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3] # 网格体积
            
            # 查找网格索引
            point_tuple = tuple(point)
            if point_tuple in coord_map:
                ide = coord_map[point_tuple]
                
                # 弥散公式: f(x) += F(X_l) * w(X_l - x)
                # 注意：这里计算的是网格上的力分布值，物理上对应 f(x)
                f_euler[ide] += F_l * modified_w[j] * V_lag / Delta_V
                
                # 记录该网格的体积 (假设同一网格点的Delta_V在不同支持域中是一致的)
                if grid_volumes[ide] == 0:
                    grid_volumes[ide] = Delta_V
            else:
                # 理论上不应发生，除非精度问题
                pass

    # 5. 计算欧拉侧的总力和总力矩
    # Force_grid = sum(f(x) * dv)
    # Torque_grid = sum((x x f(x)) * dv)
    
    # 只计算受到影响的网格点（体积不为0）
    valid_mask = grid_volumes > 0
    valid_points = eulerian_points[valid_mask]
    valid_forces = f_euler[valid_mask]
    valid_volumes = grid_volumes[valid_mask].reshape(-1, 1) # reshape for broadcasting
    
    # 欧拉总力
    total_force_euler = np.sum(valid_forces * valid_volumes, axis=0)
    
    # 欧拉总力矩
    torques_euler = np.cross(valid_points - center, valid_forces)
    total_torque_euler = np.sum(torques_euler * valid_volumes, axis=0)
    
    print(f"Eulerian Total Force: {total_force_euler}")
    print(f"Eulerian Total Torque: {total_torque_euler}")

    # 6. 计算误差
    force_diff = np.abs(total_force_lag - total_force_euler)
    force_rel_error = np.linalg.norm(force_diff) / np.linalg.norm(total_force_lag)
    
    torque_diff = np.abs(total_torque_lag - total_torque_euler)
    torque_rel_error = np.linalg.norm(torque_diff) / np.linalg.norm(total_torque_lag)
    
    print("-" * 30)
    print(f"Force Conservation Relative Error: {force_rel_error:.6e}")
    print(f"Torque Conservation Relative Error: {torque_rel_error:.6e}")
    
    return force_rel_error, torque_rel_error