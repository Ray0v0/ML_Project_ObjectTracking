import matplotlib.pyplot as plt
import numpy as np
class PathAnalyzer:
    def __init__(self):
        self.pose_to_follow_history = []  # 存储目标路径
        self.pose_follow_history = []     # 存储跟随路径
    def update_path(self, info):
        self.pose_to_follow_history.append([info[0].location.x, info[0].location.y])
        self.pose_follow_history.append([info[1].location.x, info[1].location.y])
    def save_trajectory_plot(self):
        plt.figure(figsize=(10, 10))
        # 将历史数据转换为numpy数组以便绘图
        target_path = np.array(self.pose_to_follow_history)
        follow_path = np.array(self.pose_follow_history)
        
        # 绘制两条路径
        plt.plot(target_path[:, 0], target_path[:, 1], 'r-', label='Target Path')
        plt.plot(follow_path[:, 0], follow_path[:, 1], 'b--', label='Following Path')
        
        plt.title('Trajectory Comparison')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plt.savefig(f'analysis\\trajectory_plot_{len(self.pose_to_follow_history)}.png')
        plt.close()
