import numpy as np
from volume_engine import VolumeCalculator

# 10x10 square pyramid test:
# Base 10x10 = 100 m2 at z=198.0
# Peak at center (5, 5, 201.0) -> height dh = 3.0 m
# Expected pyramid volume = 1/3 * Area * height = 1/3 * 100 * 3 = 100.0 m3
boundary = np.array([
    [439000.0, 2281300.0, 198.0],
    [439010.0, 2281300.0, 198.0],
    [439010.0, 2281310.0, 198.0],
    [439000.0, 2281310.0, 198.0]
])

bottom = np.array([
    [439005.0, 2281305.0, 198.0]
])

top = np.array([
    [439005.0, 2281305.0, 201.0]
])

calc = VolumeCalculator(top_points=top, bottom_points=bottom, boundary_points=boundary, grid_resolution=0.05)
res = calc.calculate()

print("--- Test Results ---")
print(f"Fill Volume (Насыпь): {res['v_fill']:.3f} m3")
print(f"Cut Volume (Выемка): {res['v_cut']:.3f} m3")
print(f"Net Volume (Общий объем): {res['v_net']:.3f} m3")
print(f"2D Area (Площадь основания): {res['area_2d']:.2f} m2")
print(f"Top 3D Area: {res['top_area_3d']:.2f} m2")
print(f"Bottom 3D Area: {res['bot_area_3d']:.2f} m2")
print(f"Avg Thickness (Средняя мощность): {res['avg_thickness']:.3f} m")
