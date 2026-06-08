import numpy as np
import matplotlib.pyplot as plt
from src import SI_generated, window, visual, error, mapping
import sys
import time

def main():

    # 设置测试数据参数
    sx = 1.9
    sy = 2.15
    sz = 2.15
    Lx = 0.1875
    Ly = 0.1875
    Lz = 0.1875
    nxc = 64 # the number of grid cell
    nyc = 64
    nzc = 64
    readpc = 0
    center = (2,2.25,2.25)

    eulerian_points, Ne, lagrangian_points, nearest_grid_points, delta_I, eta_I, theta_I, all_S_I, V_lag = SI_generated.generate_grid(
        sx, sy, sz, Lx, Ly, Lz, nxc, nyc, nzc, center, readpc
    )

    # print(f"拉格朗日点数目:{Ne}", area, V_lag)
    visual.visualize_results(
        lagrangian_points, nearest_grid_points,
        delta_I, eta_I, theta_I, all_S_I,
        target_idx=0  # 选择要可视化的拉格朗日点索引
    )

    start_time = time.time()

    all_modified_w = window.compute_all_modified_window_functions(
            all_S_I, lagrangian_points, delta_I, eta_I, theta_I, V_lag
    )

    solve_time = time.time() - start_time
    print(f"求解完成，耗时: {solve_time:.2f}秒")

    # 计算误差
    volumn_error = error.compute_error_volume(lagrangian_points, all_S_I, all_modified_w, V_lag)
    force_rel_error, torque_rel_error = error.compute_conservation_check(center, eulerian_points,lagrangian_points, all_S_I, all_modified_w, V_lag)
    # infinity_norm_error = error.compute_error(lagrangian_points, all_S_I, all_modified_w, V_lag)
    # print(f"无穷范数误差: {infinity_norm_error:.6e}")
        
    try:
        # 提取网格坐标
        print("提取网格坐标...")
        x_coords, y_coords, z_coords = mapping.extract_grid_coordinates(eulerian_points)
        print(f"网格尺寸: {len(x_coords)} × {len(y_coords)} × {len(z_coords)}")
        
        # 构建三大映射
        print("构建拉格朗日点ID到坐标映射...")
        id_to_coord_map = mapping.build_lagrangian_id_to_coord_map(lagrangian_points)

        print("构建拉格朗日到欧拉映射 (全部数据)")
        new_lag_to_eul_map = mapping.build_lag_to_eul_map(
            lagrangian_points, all_S_I, all_modified_w,
            x_coords, y_coords, z_coords,
            sx, sy, sz, Lx, Ly, Lz, nxc, nyc, nzc, V_lag
        )

        # txt文件
        print("保存txt文件")
        mapping.save_mappings_txt(id_to_coord_map, new_lag_to_eul_map, 'rkpm_mappings')
                    
    except Exception as e:
        print(f"✗ 映射构建过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"程序运行时间：{end - start:.4f} 秒")
