import numpy as np

# 一维窗口函数
# 输入参数：
#   r: 浮点数，归一化距离
# 输出：
#   返回窗口函数值
def window_function_d(r):
    abs_r = np.abs(r)
    if 0.5 <= abs_r <= 1.5:
        return (1/6) * (5 - 3*abs_r - np.sqrt(-3*(1-abs_r)**2 + 1))
    elif abs_r <= 0.5:
        return (1/3) * (1 + np.sqrt(-3*r**2 + 1))
    else:
        return 0

# 计算支持域内窗口函数矩阵
# 输入参数：
#   S_I: 当前拉格朗日点的支持域内欧拉点及体积信息
#   lagrangian_points: 当前拉格朗日点坐标 (3,)
#   delta_I, eta_I, theta_I: 当前拉格朗日点的支持域参数
# 输出：
#   m_ab_matrix: (10,10) 的窗口函数矩阵
def compute_m_ab_matrix(S_I, lagrangian_points, delta_I, eta_I, theta_I, V_lag):
    x_i, y_j, z_k = lagrangian_points
    m_ab_matrix = np.zeros((10, 10))

    for x_mn, y_mn, z_mn, Delta_V in S_I:
        delta_x = (x_mn - x_i) / delta_I
        delta_y = (y_mn - y_j) / eta_I
        delta_z = (z_mn - z_k) / theta_I

        dis_x = x_mn - x_i
        dis_y = y_mn - y_j
        dis_z = z_mn - z_k

        w_total = window_function_d(delta_x) * window_function_d(delta_y) * window_function_d(delta_z) * Delta_V / V_lag

        # 计算矩阵的每个元素
        m_ab_matrix[0, 0] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 1] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 2] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 3] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[0, 4] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 5] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[0, 6] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[0, 7] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 8] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[0, 9] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**2.0) * w_total

        m_ab_matrix[1, 0] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 1] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 2] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 3] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[1, 4] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 5] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[1, 6] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[1, 7] += (dis_x**3.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 8] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[1, 9] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**2.0) * w_total

        m_ab_matrix[2, 0] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 1] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 2] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 3] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[2, 4] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 5] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[2, 6] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[2, 7] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 8] += (dis_x**0.0) * (dis_y**3.0) * (dis_z**0.0) * w_total
        m_ab_matrix[2, 9] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**2.0) * w_total

        m_ab_matrix[3, 0] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 1] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 2] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 3] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[3, 4] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 5] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[3, 6] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[3, 7] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 8] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[3, 9] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**3.0) * w_total

        m_ab_matrix[4, 0] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 1] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 2] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 3] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[4, 4] += (dis_x**2.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 5] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[4, 6] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[4, 7] += (dis_x**3.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 8] += (dis_x**1.0) * (dis_y**3.0) * (dis_z**0.0) * w_total
        m_ab_matrix[4, 9] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**2.0) * w_total

        m_ab_matrix[5, 0] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 1] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 2] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 3] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[5, 4] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 5] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**2.0) * w_total
        m_ab_matrix[5, 6] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[5, 7] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 8] += (dis_x**0.0) * (dis_y**3.0) * (dis_z**1.0) * w_total
        m_ab_matrix[5, 9] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**3.0) * w_total

        m_ab_matrix[6, 0] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 1] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 2] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 3] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[6, 4] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 5] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[6, 6] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[6, 7] += (dis_x**3.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 8] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[6, 9] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**3.0) * w_total

        m_ab_matrix[7, 0] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 1] += (dis_x**3.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 2] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 3] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[7, 4] += (dis_x**3.0) * (dis_y**1.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 5] += (dis_x**2.0) * (dis_y**1.0) * (dis_z**1.0) * w_total
        m_ab_matrix[7, 6] += (dis_x**3.0) * (dis_y**0.0) * (dis_z**1.0) * w_total
        m_ab_matrix[7, 7] += (dis_x**4.0) * (dis_y**0.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 8] += (dis_x**2.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[7, 9] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**2.0) * w_total

        m_ab_matrix[8, 0] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 1] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 2] += (dis_x**0.0) * (dis_y**3.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 3] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[8, 4] += (dis_x**1.0) * (dis_y**3.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 5] += (dis_x**0.0) * (dis_y**3.0) * (dis_z**1.0) * w_total
        m_ab_matrix[8, 6] += (dis_x**1.0) * (dis_y**2.0) * (dis_z**1.0) * w_total
        m_ab_matrix[8, 7] += (dis_x**2.0) * (dis_y**2.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 8] += (dis_x**0.0) * (dis_y**4.0) * (dis_z**0.0) * w_total
        m_ab_matrix[8, 9] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**2.0) * w_total

        m_ab_matrix[9, 0] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 1] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 2] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 3] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**3.0) * w_total
        m_ab_matrix[9, 4] += (dis_x**1.0) * (dis_y**1.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 5] += (dis_x**0.0) * (dis_y**1.0) * (dis_z**3.0) * w_total
        m_ab_matrix[9, 6] += (dis_x**1.0) * (dis_y**0.0) * (dis_z**3.0) * w_total
        m_ab_matrix[9, 7] += (dis_x**2.0) * (dis_y**0.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 8] += (dis_x**0.0) * (dis_y**2.0) * (dis_z**2.0) * w_total
        m_ab_matrix[9, 9] += (dis_x**0.0) * (dis_y**0.0) * (dis_z**4.0) * w_total

    return m_ab_matrix

# 计算修正系数d_I
# 输入参数：
#   M_I: (10,10) 的窗口函数矩阵
# 输出：
#   d_I: (10,) 的修正系数数组
def compute_b_I(M_I):

    e_1 = np.zeros(10)
    e_1[0] = 1

    d_I = np.linalg.solve(M_I, e_1)
    # try:
    #     d_I = np.linalg.solve(M_I, e_1)
    # except np.linalg.LinAlgError:
    #     d_I = np.linalg.lstsq(M_I, e_1, rcond=None)[0]

    return d_I

# 计算修正窗口函数
# 输入参数：
#   S_I: 当前拉格朗日点的支持域内欧拉点及体积信息
#   lagrangian_point: 当前拉格朗日点坐标 (3,)
#   d_I: 修正系数数组
#   delta, eta, theta: 当前拉格朗日点的支持域参数
# 输出：
#   modified_w_values: 修正后的窗口函数值列表
def modified_window_function(S_I, lagrangian_point, d_I, delta, eta, theta, V_lag):

    x_i, y_j, z_k = lagrangian_point

    modified_w_values = []
    integral = 0.0

    # 退回三点
    # d_I = np.zeros(10)
    # d_I[0] = 1

    for x_mn, y_mn, z_mn, Delta_V in S_I:
        delta_x = (x_mn - x_i) / delta
        delta_y = (y_mn - y_j) / eta
        delta_z = (z_mn - z_k) / theta

        dis_x = x_mn - x_i
        dis_y = y_mn - y_j
        dis_z = z_mn - z_k

        # 计算窗函数
        w_total = window_function_d(delta_x) * window_function_d(delta_y) * window_function_d(delta_z) * Delta_V / V_lag

        # 计算修正窗函数
        modified_w = d_I[0] * w_total + \
                     d_I[1] * dis_x * w_total + \
                     d_I[2] * dis_y * w_total + \
                     d_I[3] * dis_z * w_total + \
                     d_I[4] * dis_x * dis_y * w_total + \
                     d_I[5] * dis_y * dis_z * w_total + \
                     d_I[6] * dis_z * dis_x * w_total + \
                     d_I[7] * dis_x**2 * w_total + \
                     d_I[8] * dis_y**2 * w_total + \
                     d_I[9] * dis_z**2 * w_total

        modified_w_values.append(modified_w)

        integral += modified_w

    # print(f"integral",integral,np.sum(np.array(modified_w_values)))   

    return modified_w_values

# 计算所有拉格朗日点的修正窗口函数
# 输入参数：
#   all_S_I: 所有拉格朗日点的支持域信息
#   lagrangian_points: 所有拉格朗日点坐标 (Ne, 3)
#   delta_I, eta_I, theta_I: 所有拉格朗日点的支持域参数
# 输出：
#   all_modified_w: 所有拉格朗日点的修正窗口函数值列表
def compute_all_modified_window_functions(all_S_I, lagrangian_points, delta_I, eta_I, theta_I, V_lag):

    all_modified_w = []
    for idx in range(len(lagrangian_points)):
        S_I = all_S_I[idx]
        delta = delta_I[idx]
        eta = eta_I[idx]
        theta = theta_I[idx]
        lagrangian_point = lagrangian_points[idx]

        # 计算 m_ab_matrix
        M_I = compute_m_ab_matrix(S_I, lagrangian_point, delta, eta, theta, V_lag)

        # 计算 b_I 和 d_I
        d_I = compute_b_I(M_I)

        # 计算修正窗函数
        modified_w = modified_window_function(S_I, lagrangian_point, d_I, delta, eta, theta, V_lag)

        all_modified_w.append(modified_w)

    return all_modified_w