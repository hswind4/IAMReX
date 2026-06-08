import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src import SI_generated, window, error

# 三维可视化主函数
# 输入参数：
#   eulerian_points: 所有欧拉网格点坐标 (N, 3)
#   lagrangian_points: 所有拉格朗日点坐标 (Ne, 3)
#   nearest_grid_points: 每个拉格朗日点最近的欧拉点坐标 (Ne, 3)
#   delta_I, eta_I, theta_I: 每个拉格朗日点的支持域参数（一维数组）
#   all_S_I: 每个拉格朗日点的支持域内欧拉点及体积信息
#   target_idx: 需要高亮显示的拉格朗日点索引（默认0）
# 输出：
#   无返回值，直接生成并保存三维可视化图片
def visualize_results(lagrangian_points, nearest_grid_points, 
                     delta_I, eta_I, theta_I, all_S_I, target_idx=0):
    """可视化生成的结果"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 稀疏显示欧拉网格（每skip个点显示一个）
    # skip = max(1, len(eulerian_points) // 5000)
    # ax.scatter(eulerian_points[::skip, 0], eulerian_points[::skip, 1], eulerian_points[::skip, 2],
    #            c='red', alpha=1, s=10, label=f'Eulerian Grid (1/{skip} points)')
     
    # 2. 显示所有拉格朗日点
    ax.scatter(lagrangian_points[:, 0], lagrangian_points[:, 1], lagrangian_points[:, 2],
               c='blue', s=20, alpha=0.7, label=f'Lagrangian Points (n={len(lagrangian_points)})')
    
    # 3. 高亮目标点及其S_I区域
    target_lag = lagrangian_points[target_idx]
    nearest_point = nearest_grid_points[target_idx]
    
    # 目标拉格朗日点
    ax.scatter([target_lag[0]], [target_lag[1]], [target_lag[2]],
               c='red', s=20, label=f'Target Point (idx={target_idx})')
    
    # 最近欧拉网格点
    ax.scatter([nearest_point[0]], [nearest_point[1]], [nearest_point[2]],
               c='purple', s=30, marker='s', label='Nearest Eulerian Point')
    
    # 连接线
    ax.plot([target_lag[0], nearest_point[0]], 
            [target_lag[1], nearest_point[1]], 
            [target_lag[2], nearest_point[2]],
            'k--', linewidth=1, alpha=0.5)
    
    # 4. 显示S_I区域
    S_I = all_S_I[target_idx]
    if len(S_I) > 0:
        ax.scatter(S_I[:, 0], S_I[:, 1], S_I[:, 2],
                   c='green', s=30, alpha=0.5, 
                   label=f'S_I Region (n={len(S_I)})')
        
        # 绘制边界框
        delta = delta_I[target_idx]
        eta = eta_I[target_idx]
        theta = theta_I[target_idx]
        
        # 边界框顶点
        corners = np.array([
            [nearest_point[0]-1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]+1.5*theta]
        ])
        
        # 绘制边界线
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # 底面
            [4,5],[5,6],[6,7],[7,4],   # 顶面
            [0,4],[1,5],[2,6],[3,7]    # 侧面
        ]
        for edge in edges:
            ax.plot(corners[edge, 0], corners[edge, 1], corners[edge, 2],
                   'orange', linestyle='--', linewidth=1, alpha=0.7)
    
    # 图形设置
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    # ax.set_box_aspect([1, 1, 1])
    ax.axis('equal')
    ax.set_title(
        f'3D Grid Visualization\n'
        # f'Grid Size: {np.max(eulerian_points[:,0]):.1f}, '
        f'Lagrangian Points: {len(lagrangian_points)}', 
        fontsize=14
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(
        'RKPM_3D.png',  # 文件名
        dpi=300,                      # 高分辨率
    )

def PointCloud(lagrangian_points):# 读取CSV文件
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    scatter = ax.scatter(lagrangian_points[:,0], lagrangian_points[:,1], lagrangian_points[:,2], 
                         c='black',        # 使用Z值作为颜色映射
                         s=30,             # 点的大小
                         alpha=0.8,        # 透明度
                         depthshade=True)  # 深度阴影效果

    # 添加标签和标题
    ax.set_xlabel('X Axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=10)
    ax.set_title('Scatter Plot of Lagrangian Points', fontsize=14, pad=20)

    # 调整视角
    # ax.view_init(elev=25, azim=45)  # 仰角25度，|方位角45度
    ax.view_init(elev=90, azim=-90)  # x-y plane
    # ax.view_init(elev=0, azim=-90) # x-z plane

    # 添加网格
    # ax.set_box_aspect([1, 1, 1])
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    plt.tight_layout()
    plt.savefig(
        'lagrangian.png',  # 文件名
        dpi=300,                      # 高分辨率
    )