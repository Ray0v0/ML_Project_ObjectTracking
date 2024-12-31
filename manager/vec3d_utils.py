import math


def get_magnitude(vec):
    return ((vec.x ** 2) +
            (vec.y ** 2) +
            (vec.z ** 2)) ** 0.5


def get_angle(vec1, vec2):
    dot_product = get_dot_product(vec1, vec2)
    vec1_mag = get_magnitude(vec1)
    vec2_mag = get_magnitude(vec2)

    if vec1_mag == 0 or vec2_mag == 0:
        return 0

    cos_angle = dot_product / (vec1_mag * vec2_mag)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    # 计算方向，z轴默认为上方
    cross_product = (vec1.x * vec2.y - vec1.y * vec2.x)
    if cross_product > 0:
        angle_deg = - angle_deg

    return angle_deg


def get_dot_product(vec1, vec2):
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z