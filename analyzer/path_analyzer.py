import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class PathAnalyzer:
    def __init__(self):
        self.pose_to_follow_history = []  # 存储目标路径
        self.pose_follow_history = []     # 存储跟随路径
        # self.count = 0
    def update_path(self, info):
        self.pose_to_follow_history.append([info[0].location.x, info[0].location.y])
        self.pose_follow_history.append([info[1].location.x, info[1].location.y])
        # self.count += 1
        # if self.count == 20:
        #     self.save_trajectory_plot()
        #     self.count = 0
    def save_trajectory_plot(self, controller_name, ride_filename, avg_fps):
        plt.figure(figsize=(10, 10))
        # 将历史数据转换为numpy数组以便绘图
        target_path = np.array(self.pose_to_follow_history)
        follow_path = np.array(self.pose_follow_history)
        
        # 绘制两条路径
        plt.plot(target_path[:, 0], target_path[:, 1], 'r-', label='Target Path')
        plt.plot(follow_path[:, 0], follow_path[:, 1], 'b--', label='Following Path')
        
        # Calculate metrics
        metrics = self.calculate_path_metrics()
        
        # Add metrics text to the plot
        metrics_text = (
            f"Tracking Metrics:\n"
            f"Mean Error: {metrics['mean_error']:.3f}m\n"
            f"Max Error: {metrics['max_error']:.3f}m\n"
            f"Std Error: {metrics['std_error']:.3f}m\n"
            f"Path Length: {metrics['total_path_length']} points\n"
            f"Average FPS: {avg_fps:.2f}"
        )
        
        # Add text box with metrics
        plt.text(0.02, 0.98, metrics_text,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top',
                 fontfamily='monospace')
        
        plt.title(f'Trajectory Comparison-{controller_name}-{ride_filename}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        
        # 生成带时间戳的文件名基础部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_filename = f'analysis\\{controller_name}_trajectory_{ride_filename}_{timestamp}'
        
        # 保存图片
        plt.savefig(f'{base_filename}.png')
        plt.close()
        
        # 保存指标到文本文件
        with open(f'{base_filename}.txt', 'w') as f:
            f.write(f"Controller: {controller_name}\n")
            f.write(f"Ride: {ride_filename}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Tracking Metrics:\n")
            f.write(f"Mean Error: {metrics['mean_error']:.3f}m\n")
            f.write(f"Max Error: {metrics['max_error']:.3f}m\n")
            f.write(f"Std Error: {metrics['std_error']:.3f}m\n")
            f.write(f"Path Length: {metrics['total_path_length']} points\n")
            f.write(f"Average FPS: {avg_fps:.2f}")

    def calculate_path_metrics(self):
        """计算路径差异的关键指标"""
        target_path = np.array(self.pose_to_follow_history)
        follow_path = np.array(self.pose_follow_history)
        
        # 确保两个路径长度相同
        min_length = min(len(target_path), len(follow_path))
        target_path = target_path[:min_length]
        follow_path = follow_path[:min_length]
        
        # 计算每个点之间的欧氏距离
        distances = np.sqrt(np.sum((follow_path - target_path) ** 2, axis=1))
        
        metrics = {
            'mean_error': np.mean(distances),           # 平均跟踪误差
            'max_error': np.max(distances),            # 最大跟踪误差
            'std_error': np.std(distances),            # 误差标准差
            'total_path_length': len(distances)        # 路径总长度
        }
        
        return metrics
