import numpy as np
from PIL import Image, ImageOps
import math
import random

H, W = 2000, 2000
imgMat = np.zeros((H, W, 3), dtype=np.uint8)
z_buf = np.full((H, W), np.inf, dtype=np.float64)
# упростил код относительно прошлой лабы

# загрузка obj
verts, faces = [], []
with open("model_1.obj") as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append([float(x), float(y), float(z)])
        elif line.startswith("f "):
            parts = line.split()
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:]])

# поворот и смещение модели
def rotate_and_translate(v):
    a, b, g = math.radians(30), math.radians(-150), math.radians(0)

    Rx = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), math.sin(g), 0], [-math.sin(g), math.cos(g), 0], [0, 0, 1]])

    R = Rx @ Ry @ Rz
    t = np.array([0.0, -0.04, 0.1])  # отодвигаем от камеры

    return [R @ np.array(p) + t for p in v]

verts = rotate_and_translate(verts)

# барицентрические координаты
def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)

    if denom == 0:
        return -1, -1, -1

    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
    l2 = 1 - l0 - l1

    return l0, l1, l2

# проективное преобразование
def project(x, y, z, a=800):
    u = (a * x) / z + W / 2
    v = (a * y) / z + H / 2
    return u, v

# отрисовка треугольника
def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    u0, v0 = project(x0, y0, z0)
    u1, v1 = project(x1, y1, z1)
    u2, v2 = project(x2, y2, z2)

    xmin = max(0, math.floor(min(u0, u1, u2)))
    xmax = min(W - 1, math.ceil(max(u0, u1, u2)))
    ymin = max(0, math.floor(min(v0, v1, v2)))
    ymax = min(H - 1, math.ceil(max(v0, v1, v2)))

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            l0, l1, l2 = barycentric(x, y, u0, v0, u1, v1, u2, v2)

            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z = l0 * z0 + l1 * z1 + l2 * z2

                if z < z_buf[y, x]:
                    z_buf[y, x] = z
                    imgMat[y, x] = color

# основной цикл отрисовки
for f in faces:
    x0, y0, z0 = verts[f[0]]
    x1, y1, z1 = verts[f[1]]
    x2, y2, z2 = verts[f[2]]

    n = np.cross(np.subtract(verts[f[1]], verts[f[2]]),
                 np.subtract(verts[f[1]], verts[f[0]]))
    n = n / np.linalg.norm(n)
    light = np.array([0, 0, 1])
    c = np.dot(n, light)

    if c > 0:  # отсечение нелицевых граней
        continue

    intensity = int(-255 * c)
    color = [intensity, intensity, intensity]

    draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, color)

# сохраняем изображение
img = Image.fromarray(imgMat)
img = ImageOps.flip(img)
img.save("image.png")
