import numpy as np
import matplotlib.pyplot as plt
from models.Sphere import Sphere
from models.Plane import Plane
from models.Ray import Ray

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(l, n):
    return l - 2 * np.dot(l, n) * n


def trace(ray, scene, depth=0):
    if depth > 5:
        return np.array([0, 0, 0])

    closest_t = np.inf
    hit_object = None
    hit_point = None
    normal = None

    for obj in scene['objects']:
        hit, t = obj.intersect(ray)
        if hit and t < closest_t:
            closest_t = t
            hit_object = obj
            hit_point = ray.origin + ray.direction * t
            if isinstance(obj, Sphere):
                normal = normalize(hit_point - obj.center)
            elif isinstance(obj, Plane):
                normal = obj.normal

    if hit_object is not None:
        ambient = 0.1
        to_light = normalize(scene['light']['position'] - hit_point)
        light_ray = Ray(hit_point + normal * 0.001, to_light)
        light_intersection = any(obj.intersect(light_ray)[0] for obj in scene['objects'] if obj is not hit_object)

        if light_intersection:
            return hit_object.color * ambient  # In shadow

        diffuse = max(np.dot(normal, to_light), 0)
        color = hit_object.color * (ambient + (1 - ambient) * diffuse)
        return np.clip(color, 0, 1)
    return np.array([0, 0, 0])

def render(film, camera, scene):
    height, width, _ = film.shape
    for i in range(width):
        for j in range(height):
            xn = (2 * ((i + 0.5) / width) - 1) * np.tan(camera['fov'] / 2) * width / height
            yn = (1 - 2 * ((j + 0.5) / height)) * np.tan(camera['fov'] / 2)
            ray = Ray(camera['position'], [xn, yn, -1])
            color = trace(ray, scene)
            film[j, i] = color

camera = {'position': [0, 0, 0], 'fov': np.pi/4}
scene = {
    'objects': [
        Sphere([0, 0, -5], 1, [1, 0, 0]),  # Esfera vermelha
        # Sphere([2, 1, -7], 1, [0, 1, 0]),  # Esfera verde
        Plane([0, -1, 0], [0, 1, 0], [1, 1, 1])  # Chão branco
    ],
    'light': {'position': [0, 10, 0], 'color': [1, 1, 1]}  # Luz vindo de cima, x,y,z
}

film = np.zeros((500, 500, 3))  # 500x500 image with 3 color channels

render(film, camera, scene)

plt.imshow(film)
plt.axis('off')
plt.show()