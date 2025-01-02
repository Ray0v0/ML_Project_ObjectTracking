import numpy as np

from ultralytics import YOLO

from dto.daf_info import DAFInfo


class DistanceAndAnglePerceiver:
    """标准感知器类"""

    def __init__(self):
        # 加载预训练的YOLOv8模型
        self.yolo_model = YOLO('model\\yolov8n.pt')

        # 更新相机参数以提高准确性
        self.camera_matrix = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ])

        self.min_detection_confidence = 0.6  # 提高检测置信度阈值



    def get_box_from_image(self, camera_image):
        """处理CARLA相机输入，只检测最相关的一辆车"""

        if camera_image is None:
            return None

        image = camera_image.copy()
        results = self.yolo_model(image, conf=self.min_detection_confidence)
        best_box = None

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            # 只关注车辆类别
            vehicle_classes = [2]  # 只检测轿车类别

            # 找到最相关的一辆车
            best_score = -1

            for box in boxes:
                if int(box.cls) in vehicle_classes:
                    box_data = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)

                    # 计算框的中心点和面积
                    center_x = (box_data[0] + box_data[2]) / 2
                    center_y = (box_data[1] + box_data[3]) / 2
                    box_area = (box_data[2] - box_data[0]) * (box_data[3] - box_data[1])

                    # 计算到图像中心的距离
                    image_center_x = image.shape[1] / 2
                    image_center_y = image.shape[0] / 2
                    center_dist = np.sqrt((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2)

                    # 评分标准：优先选择图像中心、较大且置信度高的车辆
                    score = confidence * (1 - center_dist / 1000) * (box_area / (image.shape[0] * image.shape[1]))

                    if score > best_score:
                        best_score = score
                        best_box = box_data

        return best_box


    def get_distance_and_angle_from_box(self, box, IMAGE_HEIGHT=600, IMAGE_WIDTH=900, IMAGE_FOV=90):
        if box is None:
            return None, None

        # 改进的距离估算
        box_height = box[3] - box[1]
        box_width = box[2] - box[0]
        aspect_ratio = box_width / box_height # ?

        # 考虑实际车辆尺寸和透视效果
        VEHICLE_HEIGHT = 1.5  # 米
        VEHICLE_WIDTH = 1.8  # 米

        # 使用高度和宽度的组合来估算距离
        distance_by_height = (VEHICLE_HEIGHT * self.camera_matrix[1, 1]) / box_height
        distance_by_width = (VEHICLE_WIDTH * self.camera_matrix[0, 0]) / box_width
        distance = (distance_by_height + distance_by_width) / 2

        # 计算相对位置
        box_center_x = (box[0] + box[2]) / 2
        image_center_x = IMAGE_WIDTH / 2

        angle = (box_center_x - image_center_x) * IMAGE_FOV / IMAGE_WIDTH
        return distance, angle






    # def _draw_debug_info(self, image, box, distance, heading):
    #     """绘制调试信息"""
    #     cv2.rectangle(image,
    #                   (int(box[0]), int(box[1])),
    #                   (int(box[2]), int(box[3])),
    #                   (0, 255, 0), 2)
    #
    #     info_text = [
    #         f'Distance: {distance:.2f}m',
    #         f'Heading: {np.degrees(heading):.1f}deg',
    #     ]
    #
    #     y_offset = int(box[1] - 10)
    #     for text in info_text:
    #         cv2.putText(image, text,
    #                     (int(box[0]), y_offset),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.9, (0, 255, 0), 2)
    #         y_offset -= 25
    #
    #     cv2.imshow('Vehicle Detection', image)
    #     cv2.waitKey(1)



    def perceive(self, velocity_follow=None, pose_follow=None, map=None, camera_image=None):

        # 将image.raw_data转化为(H, W, 3)的numpy数组
        image_numpy = np.frombuffer(camera_image.raw_data, dtype=np.uint8)
        image_numpy = image_numpy.reshape((camera_image.height, camera_image.width, 4))
        image_numpy = image_numpy[:, :, :3]

        # 使用yolo识别前车在图像中的位置
        box = self.get_box_from_image(image_numpy)
        if box is None:
            return DAFInfo(0, 0, False)
        else:
            distance, angle = self.get_distance_and_angle_from_box(box)
            return DAFInfo(distance, angle)