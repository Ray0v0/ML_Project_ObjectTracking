import carla
import queue

# Carla连接助手
class SyncCarlaManager(object):
    # 在初始化时需要传入所需的传感器，用于在每个tick()中获取传感器信息
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self.frame = None
        self._queues = []
        self._settings = None

    # 在with ... as sync_mode时自动调用
    # 至于为什么要make_queue，为什么是这么写的，官方给的，我也不知道^_^
    def __enter__(self):
        self.settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds,
        ))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        # 加载当前世界信息
        make_queue(self.world.on_tick)
        # 加载各传感器信息
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self.settings)

    # 获取当前tick的世界信息与传感器信息
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        return data

    # 收集数据的具体方法
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # 由于数据获取可能存在延迟
            # 确保获取到的是当前帧的数据
            if data.frame == self.frame:
                return data