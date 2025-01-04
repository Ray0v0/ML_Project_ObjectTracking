import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class RegressionModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=2):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def read_data(file_path):
    boxes = []
    distances = []
    angles = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            box = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
            distance = float(parts[4])
            angle = float(parts[5])
            boxes.append(box)
            distances.append(distance)
            angles.append(angle)
    boxes = np.array(boxes)
    distances = np.array(distances).reshape(-1, 1)
    angles = np.array(angles).reshape(-1, 1)
    labels = np.hstack((distances, angles))  # 将distance和angle合并为一个标签数组
    return boxes, labels



def main():
    # 读取和预处理数据
    X, y = read_data('../data/box_to_distance_and_angle.txt')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 实例化模型、定义损失函数和优化器
    model = RegressionModel(input_size=4, hidden_size=32, output_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss = None

    # 训练循环
    num_epochs = 120
    batch_size = 16  # 你可以根据需要调整批次大小
    for epoch in range(num_epochs):
        # 注意：在实际应用中，你应该使用DataLoader来更有效地管理数据批次
        # 这里为了简单起见，我们使用一个简单的循环来迭代数据
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch后打印损失（可选）
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()

    # 创建一个列表来存储预测值
    predictions = []

    # 使用测试数据进行前向传播
    with torch.no_grad():  # 禁用梯度计算以加速推理并减少内存消耗
        for batch_X in DataLoader(X_test_tensor, batch_size=batch_size, shuffle=False):
            batch_predictions = model(batch_X)
            predictions.append(batch_predictions.cpu().numpy())

    predictions = np.vstack(predictions)
    mse = np.mean((predictions - y_test_tensor.numpy()) ** 2, axis=0)
    print(f'Mean Squared Error (MSE) on test set: {mse}')

    for i in range(len(predictions)):
        print(predictions[i], y_test_tensor.numpy()[i])

    torch.save(model.state_dict(), '../model/box_to_distance_and_angle_model.pth')


if __name__ == '__main__':
    main()