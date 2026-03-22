import numpy as np
import matplotlib.pyplot as plt

def simple_hdr_merge(frame_long, frame_short, ratio=10.0, threshold=1000):
    # Создаем пустой массив для результата
    merged_frame = np.zeros_like(frame_long, dtype=np.float32)
    
    # Проходим по всем пикселям (в numpy это делается векторно)
    # Маска для пикселей, которые НЕ пересвечены на длинной выдержке
    mask_good_long = frame_long < threshold
    
    # 1. Там, где длинная выдержка хорошая — берем её
    merged_frame[mask_good_long] = frame_long[mask_good_long]
    
    # 2. Там, где длинная пересвечена — берем короткую и умножаем на ratio
    # (восстанавливаем яркость)
    merged_frame[~mask_good_long] = frame_short[~mask_good_long] * ratio
    
    return merged_frame

class VirtualScene:
    """
    Моделирует идеальный мир с бесконечной точностью (float).
    Генерирует тестовый сигнал (градиент света).
    """
    def generate_ramp(self, max_lux, num_points=1000):
        # Создаем линейно нарастающий сигнал освещенности от 0 до max_lux
        # Это аналог "пирамидки" или градиента, упомянутого в разговоре [cite: 153, 417]
        lux_array = np.linspace(0, max_lux, num_points)
        return lux_array

class VirtualSensor:
    """
    Моделирует физический сенсор с ограничениями.
    """
    def __init__(self, bit_depth=10, saturation_level=1023):
        # Сенсор ограничен 10 битами (0..1023) [cite: 183]
        self.max_val = saturation_level
        self.bit_depth = bit_depth

    def capture(self, scene_lux, exposure_time, gain=1.0):
        """
        Превращает свет (Lux) в цифровой сигнал (DN - Digital Number).
        Формула: Signal = Light * Exposure * Gain
        """
        # 1. Формирование идеального сигнала (в мире float/double) 
        raw_signal = scene_lux * exposure_time * gain
        
        # 2. Имитация насыщения (Clipping). Всё, что выше 1023 — отрезается [cite: 213]
        clipped_signal = np.clip(raw_signal, 0, self.max_val)
        
        # 3. Квантование (Дискретизация). Перевод из float в int [cite: 183]
        # Отбрасываем дробную часть (floor)
        digital_signal = np.floor(clipped_signal).astype(int)
        
        return digital_signal

# --- ЗАПУСК МОДЕЛИРОВАНИЯ ---

# 1. Настройка параметров
MAX_SCENE_LUX = 50000.0  # Очень яркий свет (например, блик)
SENSOR_BITS = 10         # 10-битный сенсор
EXPOSURE_LONG = 1.0      # Длинная выдержка (условные единицы)
EXPOSURE_SHORT = 0.1     # Короткая выдержка (1/10 от длинной) 

# 2. Инициализация объектов
scene = VirtualScene()
sensor = VirtualSensor(bit_depth=SENSOR_BITS)

# 3. Генерация сцены (Идеальный свет)
# Генерируем "пандус" света, чтобы проверить динамический диапазон
light_data = scene.generate_ramp(MAX_SCENE_LUX)

# 4. "Съемка" сенсором (Staggered HDR)
# Получаем два кадра: один для темных участков, другой для светлых
frame_long = sensor.capture(light_data, exposure_time=EXPOSURE_LONG)
frame_short = sensor.capture(light_data, exposure_time=EXPOSURE_SHORT)

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(12, 8))

# График 1: Исходный свет
plt.subplot(3, 1, 1)
plt.plot(light_data, color='orange', linestyle='--')
plt.title('1. Входной сигнал сцены (Идеальные Люксы)')
plt.ylabel('Освещенность (Lux)')
plt.grid(True)

# График 2: Длинная выдержка (Long Exposure)
plt.subplot(3, 1, 2)
plt.plot(frame_long, color='green')
plt.axhline(y=1023, color='r', linestyle=':', label='Предел насыщеня (1023)')
plt.title(f'2. Сенсор: Длинная выдержка (Exp={EXPOSURE_LONG})')
plt.ylabel('Цифровой сигнал (0-1023)')
plt.legend()
plt.grid(True)
# Примечание: Здесь видно, как сигнал быстро достигает "потолка" 

# График 3: Короткая выдержка (Short Exposure)
plt.subplot(3, 1, 3)
plt.plot(frame_short, color='blue')
plt.axhline(y=1023, color='r', linestyle=':', label='Предел насыщеня (1023)')
plt.title(f'3. Сенсор: Короткая выдержка (Exp={EXPOSURE_SHORT})')
plt.ylabel('Цифровой сигнал (0-1023)')
plt.xlabel('Позиция на сенсоре (пиксели)')
plt.legend()
plt.grid(True)
# Примечание: Здесь сигнал растет медленнее, позволяя различить яркие детали, 
# которые "сгорели" на длинной выдержке.

plt.tight_layout()
plt.show()

# Начальные параметры
current_exposure = 1.0
target_brightness = 500  # Мы хотим, чтобы средняя яркость была в середине диапазона (0-1023)
history = []

# Цикл по времени (например, 50 кадров)
for t in range(50):
    # 1. Меняем яркость мира (симуляция: солнце вышло из-за туч)
    # Пусть яркость растет от 1000 до 50000 люкс
    current_light_level = 1000 + t * 1000 
    scene_data = scene.generate_ramp(current_light_level)
    
    # 2. Снимаем кадры с ТЕКУЩЕЙ экспозицией
    # Важно: соотношение long/short всегда фиксировано (например, 1/10) [cite: 34]
    exp_long = current_exposure
    exp_short = current_exposure / 10.0 
    
    f_long = sensor.capture(scene_data, exp_long)
    f_short = sensor.capture(scene_data, exp_short)
    
    # 3. Сшиваем HDR (чтобы оценить реальную яркость сцены)
    hdr_image = simple_hdr_merge(f_long, f_short, ratio=10.0)
    
    # 4. Логика Автоматики (Регулятор)
    # Считаем среднюю яркость по кадру
    avg_val = np.mean(f_long) # Ориентируемся по длинному кадру (для простоты)
    
    # Простейший пропорциональный регулятор
    if avg_val > 800: # Слишком ярко, уходим в насыщение
        current_exposure *= 0.8 # Уменьшаем на 20%
    elif avg_val < 300: # Слишком темно
        current_exposure *= 1.2 # Увеличиваем на 20%
        
    print(f"Кадр {t}: Свет={current_light_level}, Эксп={current_exposure:.4f}, Среднее={avg_val:.1f}")
    history.append(avg_val)

# Постройте график history, чтобы увидеть, как ваша автоматика "держит" яркость
plt.figure()
plt.plot(history)
plt.title("Работа автоэкспозиции (стабилизация яркости)")
plt.show()