import numpy as np
import matplotlib.pyplot as plt
from src import SI_generated, window, visual, error, mapping
import sys
import time
import argparse
from pathlib import Path

# AMReX inputs 文件中 max_level 的键名。
MAX_LEVEL_KEY = "amr.max_level"
DEFAULT_INPUTS = "inputs.3d.flow_past_ellipsoid"

# 解析 AMReX 风格的 inputs 文件（key = value，# 行内/整行注释，值可为空格分隔列表）
def parse_inputs(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()  # 去掉行内注释
            if not line or '=' not in line:
                continue
            key, _, val = line.partition('=')
            params[key.strip()] = val.strip()
    return params


# 取空格分隔的浮点列表
def get_floats(d, key):
    if key not in d:
        raise KeyError(f"inputs 文件中缺少参数: {key}")
    return [float(x) for x in d[key].split()]


# 取整型（取第一个 token）
def get_int(d, key):
    if key not in d:
        raise KeyError(f"inputs 文件中缺少参数: {key}")
    return int(d[key].split()[0])


def main(inputs_path, geometry_file, body_frame=False, angle=0.0):

    # --- 从 inputs 文件读取域/网格参数 ---
    # 最细网格尺寸：第 level 层 dx = (prob_hi-prob_lo)/(n_cell*2**level)，
    # level=max_level 即最细层（amr.ref_ratio=2，每层细化 2 倍）。每层网格独立编号，
    # 索引 = floor((x - prob_lo) / dx_level)。
    params = parse_inputs(inputs_path)
    prob_lo = np.array(get_floats(params, "geometry.prob_lo"))
    prob_hi = np.array(get_floats(params, "geometry.prob_hi"))
    n_cell = np.array(get_floats(params, "amr.n_cell"))
    max_level = get_int(params, MAX_LEVEL_KEY)
    dx_finest = (prob_hi - prob_lo) / (n_cell * (2 ** max_level))
    print(f"[inputs] prob_lo = {prob_lo.tolist()}")
    print(f"[inputs] prob_hi = {prob_hi.tolist()}")
    print(f"[inputs] amr.n_cell = {n_cell.tolist()}, {MAX_LEVEL_KEY} = {max_level}")
    print(f"[inputs] dx_finest = {dx_finest.tolist()}")

    # --body-frame: 点云为 body frame（近原点），读 inputs 的 particle_inputs.x/y/z
    # 作几何体中心，在 generate_grid 中整体平移过去（不旋转）。
    # 默认（世界坐标）传 None，点云按世界坐标直接使用。
    if body_frame:
        center = np.array([
            get_floats(params, "particle_inputs.x")[0],
            get_floats(params, "particle_inputs.y")[0],
            get_floats(params, "particle_inputs.z")[0],
        ])
        print(f"[inputs] center (particle_inputs) = {center.tolist()}")
    else:
        center = None

    # 直接在求解器最细网格的全局坐标系下生成欧拉网格与拉格朗日点：
    # 欧拉区域由点云 min/max 包裹并向外扩 2 个最细网格，全局索引由 prob_lo 直接算出，
    # 无局部子区域、无索引偏置。
    eulerian_points, Ne, lagrangian_points, nearest_grid_points, delta_I, eta_I, theta_I, all_S_I, V_lag = SI_generated.generate_grid(
        prob_lo, prob_hi, dx_finest, geometry_file, center=center, angle=angle
    )

    # 力矩守恒校验的参考点：body-frame 用 inputs 中心；世界坐标用点云 bbox 中心
    if center is None:
        center = (lagrangian_points.min(axis=0) + lagrangian_points.max(axis=0)) / 2.0
        print(f"[inputs] center (bbox) = {center.tolist()}")

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
    force_rel_error, torque_rel_error = error.compute_conservation_check(center, eulerian_points, lagrangian_points, all_S_I, all_modified_w, V_lag)
    # infinity_norm_error = error.compute_error(lagrangian_points, all_S_I, all_modified_w, V_lag)
    # print(f"无穷范数误差: {infinity_norm_error:.6e}")

    try:
        # 构建三大映射
        print("构建拉格朗日点ID到坐标映射...")
        id_to_coord_map = mapping.build_lagrangian_id_to_coord_map(lagrangian_points)

        print("构建拉格朗日到欧拉映射 (全部数据)")
        new_lag_to_eul_map = mapping.build_lag_to_eul_map(
            lagrangian_points, all_S_I, all_modified_w,
            prob_lo, dx_finest, V_lag
        )

        # txt文件
        print("保存txt文件")
        mapping.save_mappings_txt(id_to_coord_map, new_lag_to_eul_map, 'rkpm_mappings')

    except Exception as e:
        print(f"✗ 映射构建过程出错: {e}")
        import traceback
        traceback.print_exc()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成 RKPM 权重映射")
    parser.add_argument(
        "--inputs", default=None,
        help=f"AMReX inputs 文件路径（默认: {DEFAULT_INPUTS}，相对脚本目录解析）")
    parser.add_argument(
        "--geometry", default=None,
        help="拉格朗日点云文件路径（必须提供）")
    parser.add_argument(
        "--body-frame", action="store_true", default=False,
        help="点云为 body frame（近原点）：读 inputs 的 particle_inputs.x/y/z 作中心，"
             "把点云整体平移到该中心。默认按世界坐标使用点云。")
    parser.add_argument(
        "--angle", type=float, default=0.0,
        help="绕 z 轴的旋转角度（度，默认 0，仅改变 x,y）。平移之后再旋转："
             "--body-frame 时枢轴为 particle_inputs 中心，否则为世界原点。")
    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_args()
    if args.inputs is None:
        inputs_path = (Path(__file__).parent / DEFAULT_INPUTS).resolve()
    else:
        inputs_path = Path(args.inputs)

    start = time.perf_counter()
    main(inputs_path, args.geometry, args.body_frame, args.angle)
    end = time.perf_counter()
    print(f"程序运行时间：{end - start:.4f} 秒")
