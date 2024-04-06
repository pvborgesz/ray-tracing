import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class Plane:
    def __init__(self, point, normal, color):
        self.point = np.array(point)
        self.normal = normalize(np.array(normal))
        self.color = np.array(color)

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if np.abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t >= 0:
                return True, t
        return False, None