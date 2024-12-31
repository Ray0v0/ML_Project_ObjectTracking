import time
import keyboard
import threading

from controller.traditional_controller import TraditionalController
import carla

class KeyboardStatus:
    def __init__(self):
        # 初始化键状态，所有键初始为未按下
        self.keys = {
            'w': False,
            'a': False,
            's': False,
            'd': False
        }

        # 监听键盘按下和释放事件
        keyboard.hook(self._update_key_status)

    def _update_key_status(self, event):
        """
        该方法在键盘事件发生时调用，更新键的状态。
        """
        if event.name in self.keys:
            if event.event_type == keyboard.KEY_DOWN:
                self.keys[event.name] = True
            elif event.event_type == keyboard.KEY_UP:
                self.keys[event.name] = False

    def is_pressed(self, key):
        """
        返回特定键是否被按下。
        :param key: 要检查的键 ('w', 'a', 's', 'd')
        :return: True 如果键被按下，False 否则
        """
        if key not in self.keys:
            raise ValueError("Invalid key! Valid keys are 'w', 'a', 's', 'd'.")
        return self.keys[key]

class ManualController(TraditionalController):
    def __init__(self):
        self.keyboard_status = KeyboardStatus()
        self.throttle = 0
        self.brake = 0
        self.steer = 0

    def predict_control(self, info):
        if self.keyboard_status.is_pressed('w'):
            self.throttle = 1
        else:
            self.throttle = 0

        if self.keyboard_status.is_pressed('s'):
            self.throttle = 0
            self.brake = 1
        else:
            self.brake = 0

        self.steer = 0

        if self.keyboard_status.is_pressed('a'):
            self.steer -= 1

        if self.keyboard_status.is_pressed('d'):
            self.steer += 1

        # print("\rthrottle:{}, steer:{}, brake:{}".format(self.throttle, self.steer, self.brake), end='')
        return carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brake)

# if __name__ == '__main__':
#     controller = ManualController()
#     while True:
#         controller.predict_control(None)
#         time.sleep(0.1)