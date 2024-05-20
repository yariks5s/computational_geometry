import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

canvas = None

def convex_hull(points):
    """Обчислює опуклу оболонку з набору заданих точок за домомогою Алгоритма Грехема"""
    # Конвертуємо точки в масив нампай
    if not isinstance(points, np.ndarray):
        points = np.array(points)
        
    # Соруємо лексикографічно точки
    indices = np.lexsort((points[:, 1], points[:, 0]))
    points = points[indices]

    # Випадок якщо немає точок або багато однакових точок або одна точка
    if len(points) <= 1:
        return points

    # Множення векторів OA i OB визначає кут
    # Повертає позитивне значення якщо OAB обертається проти годинникової стрілки
    # негативний для випадку коли оборот за год стрілкою і 0 коли вони колінеарні
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Будуємо нижню оболонку
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Будуємо верхню оболонку
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Конкатенуємо верхні і нижні оболонки
    return np.array(lower[:-1] + upper[:-1])

def binary_search_peak(values):
    """Find the index of the peak (maximum) using binary search."""
    left, right = 0, len(values) - 1
    while left < right:
        mid = (left + right) // 2
        if values[mid] < values[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left

def binary_search_leftmost(values):
    """Find the index of the leftmost minimum using binary search."""
    left, right = 0, len(values) - 1
    while left < right:
        mid = (left + right) // 2
        if values[mid] > values[mid - 1] and mid > 0:
            right = mid
        else:
            left = mid + 1
    return left

def binary_search_rightmost(values):
    """Find the index of the rightmost maximum using binary search."""
    left, right = 0, len(values) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if values[mid] < values[mid - 1] and mid > 0:
            right = mid - 1
        else:
            left = mid
    return left

def rotating_calipers(points):
    hull_points = convex_hull(points)
    n = len(hull_points)
    min_area = float('inf')
    best_rect = None

    for i in range(n):
        p1 = hull_points[i-1]
        p2 = hull_points[i]
        # ВИзначаємо точки для шуканого ребра та робимо з нього вектор
        edge = p2 - p1
        edge_length = np.linalg.norm(edge) # Нормалізуємо вектор 
        if edge_length == 0:
            continue
        edge_direction = edge / edge_length

         # Створюємо нормаль до вектора. Вектор і нормаль мають спільну початкову точку, але перпендикулярні
        normal = np.array([-edge_direction[1], edge_direction[0]])

        # Переходимо до локальної системи координат
        local_points = hull_points - p1
        x_coords = np.dot(local_points, edge_direction)
        y_coords = np.dot(local_points, normal)
        # Знаходимо проекції в новій системі координат

        # Знаходимо найвищу, найправішу, та найлівішу точки в локальних координатаї
        highest_point_index = np.argmax(y_coords)
        leftmost_point_index = np.argmin(x_coords)
        rightmost_point_index = np.argmax(x_coords)

        # Координати прямокутника в стандартних координатах
        top_y = y_coords[highest_point_index]
        left_x = x_coords[leftmost_point_index]
        right_x = x_coords[rightmost_point_index]

        # Площа прямокутника
        width = right_x - left_x
        height = top_y
        area = width * height

        if area < min_area:
            min_area = area
            best_rect = (p1, edge_direction, normal, left_x, right_x, 0, top_y)

    return best_rect, min_area

def plot_points(points):
    global canvas

    if len(points) < 3:
        print("Неправильна кількість точок. Треба як мінімум 3.")
        return

    best_rect, min_area = rotating_calipers(points)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='datalim')

    ax.plot(points[:,0], points[:,1], 'o')

    # Опукла оболонка
    hull_points = convex_hull(points)
    ax.plot(hull_points[:,0], hull_points[:,1], 'k-', lw=2)
    ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]), 
            np.append(hull_points[:, 1], hull_points[0, 1]), 'k-', lw=2)

    if best_rect:
        p1, edge_direction, normal, left_x, right_x, bottom_y, top_y = best_rect
        rect_points = [
            p1 + left_x * edge_direction + bottom_y * normal,
            p1 + right_x * edge_direction + bottom_y * normal,
            p1 + right_x * edge_direction + top_y * normal,
            p1 + left_x * edge_direction + top_y * normal,
            p1 + left_x * edge_direction + bottom_y * normal
        ]
        rect_points = np.array(rect_points)
        ax.plot(rect_points[:, 0], rect_points[:, 1], 'r-', lw=2)

    if canvas is not None:
        canvas.get_tk_widget().pack_forget()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    print("Координати прямокутника:")
    if best_rect:
        for i, point in enumerate(rect_points):
            print(f"Точка {i + 1}:", point)
        print("Ширина:", np.linalg.norm(rect_points[1] - rect_points[0]))
        print("Висота:", np.linalg.norm(rect_points[2] - rect_points[1]))
        print("Мінімальна площа:", min_area)

def generate_ten_points():
    points = np.random.rand(10, 2) * 100
    plot_points(points)

def generate_100k_points():
    points = np.random.rand(100000, 2) * 100000000
    plot_points(points)

def input_points():
    points = []
    while True:
        coords = simpledialog.askstring("Ввід", "Введи координати у форматі a, b. Натисність 'q' для виходу з цього вікна")
        if coords.lower() == 'q':
            break
        try:
            x, y = map(float, coords.split(','))
            points.append([x, y])
        except ValueError:
            print("Сталась помилка. Будь ласка, переконайтесь що ви ввели числа в правильному форматі")
    points = np.array(points)
    plot_points(points)

root = tk.Tk()
root.title("Лабораторна робота")

buttonA = tk.Button(root, text="Згенерувати 10 точок", command=generate_ten_points)
buttonA.pack()

buttonB = tk.Button(root, text="Згенерувати 100k точок", command=generate_100k_points)
buttonB.pack()

buttonC = tk.Button(root, text="Ручний ввід точок", command=input_points)
buttonC.pack()

buttonExit = tk.Button(root, text="Exit", command=root.quit)
buttonExit.pack()

root.mainloop()
