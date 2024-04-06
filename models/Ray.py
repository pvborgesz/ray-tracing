import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalize(np.array(direction))
