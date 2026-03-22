import numpy as np
import matplotlib.pyplot as plt

def generate_radial_gradient(size=512, max_val=50000):
    """
    Генерирует изображение "свет в конце туннеля".
    В центре очень ярко, по краям темно.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    
    # Инвертируем радиус: чем ближе к центру (0), тем ярче.
    # Используем экспоненту, чтобы имитировать резкий спад света.
    light_source = np.exp(-radius * 5) * max_val
    return light_source

def sensor_capture(image_lux, exposure):
    """
    Имитация сенсора:
    1. Умножаем свет на выдержку.
    2. Обрезаем всё, что выше 1023 (насыщение).
    3. Округляем до целых.
    """
    signal = image_lux * exposure
    signal = np.clip(signal, 0, 1023) # Жесткое ограничение 10 бит
    return signal.astype(int)

def simple_hdr_reconstruction(long_img, short_img, ratio):
    """
    Склейка двух изображений.
    """
    threshold = 1000 # Порог, где мы считаем, что длинная выдержка "ослепла"
    
    # Создаем пустую картинку
    hdr_result = np.zeros_like(long_img, dtype=float)
    
    # Маска: пиксели, которые НЕ пересвечены на длинной выдержке
    mask_ok = long_img < threshold
    
    # 1. Там, где видно детали на длинной выдержке - берем их
    hdr_result[mask_ok] = long_img[mask_ok]
    
    # 2. Там, где длинная выдержка белая (пересвет) - берем короткую и усиливаем
    hdr_result[~mask_ok] = short_img[~mask_ok] * ratio
    
    return hdr_result

# --- ПАРАМЕТРЫ ---
MAX_LIGHT = 100000.0 # Очень яркий источник света (100 тыс люкс)
RATIO = 16.0         # Разница выдержек в 16 раз
EXP_LONG = 1.0       # Длинная выдержка

# 1. Генерируем "Идеальный мир" (Сцена)
scene = generate_radial_gradient(max_val=MAX_LIGHT)

# 2. Сенсор видит мир (Два кадра)
# Длинная выдержка: чтобы видеть тени по краям
img_long = sensor_capture(scene, EXP_LONG)
# Короткая выдержка: чтобы видеть супер-яркий центр
img_short = sensor_capture(scene, EXP_LONG / RATIO)

# 3. Магия HDR (Склейка)
img_hdr = simple_hdr_reconstruction(img_long, img_short, RATIO)

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(15, 5))

# Картинка 1: Длинная выдержка (Обычная камера)
plt.subplot(1, 3, 1)
plt.imshow(img_long, cmap='gray', vmin=0, vmax=1023)
plt.title(f'Длинная выдержка\n(Центр "сгорел" - белое пятно)')
plt.colorbar(label='Яркость (0-1023)')

# Картинка 2: Короткая выдержка
plt.subplot(1, 3, 2)
plt.imshow(img_short, cmap='gray', vmin=0, vmax=1023)
plt.title(f'Короткая выдержка\n(Центр виден, края черные)')
plt.colorbar()

# Картинка 3: Результат HDR
plt.subplot(1, 3, 3)
plt.imshow(img_hdr, cmap='gray') # Здесь масштаб автоподстроится под HDR
plt.title(f'Результат HDR\n(Видно и центр, и края!)')
plt.colorbar(label='Восстановленная яркость')

plt.tight_layout()
plt.show()