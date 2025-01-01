import math

# 提供处理Vec3d的各种工具

# 取模
def get_magnitude(vec):
    return ((vec.x ** 2) +
            (vec.y ** 2) +
            (vec.z ** 2)) ** 0.5

# 获取两个向量之间的角度
# 通过余弦定理获取角度绝对值
# 以vec1为前方，z轴（0，0，1）为上方，区分左右
def get_angle(vec1, vec2):
    dot_product = get_dot_product(vec1, vec2)
    vec1_mag = get_magnitude(vec1)
    vec2_mag = get_magnitude(vec2)

    if vec1_mag == 0 or vec2_mag == 0:
        return 0

    cos_angle = dot_product / (vec1_mag * vec2_mag)

    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    cross_product = (vec1.x * vec2.y - vec1.y * vec2.x)
    if cross_product > 0:
        angle_deg = - angle_deg

    return angle_deg

# 点乘
def get_dot_product(vec1, vec2):
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z