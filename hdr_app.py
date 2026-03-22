import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. ФИЗИЧЕСКИЕ МОДЕЛИ ---

def generate_fir_tree(max_lux, num_steps=10, width_px=1000):
    """Генератор симметричной 'ёлочки' (имитация блика на полипе)."""
    wedge = np.zeros(width_px)
    step_width = width_px // (num_steps * 2)
    center = width_px // 2
    
    current_lux = max_lux
    for i in range(num_steps):
        left = max(0, center - (i + 1) * step_width)
        right = min(width_px, center + (i + 1) * step_width)
        # Заполняем симметричные ступени
        wedge[left : center - i * step_width] = current_lux
        wedge[center + i * step_width : right] = current_lux
        current_lux /= 2.0 
    return wedge

class Sensor10Bit:
    """Модель сенсора с ограничением АЦП и проверкой целых строк."""
    def __init__(self, k_sens=20.0):
        self.max_val = 1023.0
        self.k_sens = k_sens 
        
    def capture(self, scene_lux, exposure_lines, gain=1.0, lens_coeff=10.0):
        if not isinstance(exposure_lines, int):
            raise ValueError("Ошибка: Экспозиция должна быть целой строкой!")
        
        # Пересчет света в отсчеты АЦП
        signal = (scene_lux / lens_coeff / self.k_sens) * exposure_lines * gain
        return np.floor(np.clip(signal, 0, self.max_val)).astype(int)

def process_hdr_16bit(frame_long, frame_short, ratio):
    """Линейная склейка и нормировка до 16 бит (0-65535)."""
    # Склейка: если пиксель сгорел (<1000), берем из короткого кадра
    hdr_float = np.where(frame_long < 1000, frame_long, frame_short * ratio)
    
    # Нормировка по теоретическому максимуму
    max_theoretical = 1023.0 * ratio 
    normalized = np.clip(hdr_float / max_theoretical, 0, 1.0)
    
    return (normalized * 65535).astype(np.uint16)


# --- 2. ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(layout="wide", page_title="HDR Endoscopy Simulator")
st.title("🔬 Виртуальный стенд: Staggered HDR")

# Боковая панель: Ползунки параметров
with st.sidebar:
    st.header("⚙️ Динамика сцены")
    max_lux = st.slider("Макс. яркость блика (Lux)", 1000, 200000, 100000, step=5000)
    lens_coeff = st.slider("Ослабление объектива", 1.0, 32.0, 10.0, step=1.0)
    
    st.header("🎛️ Параметры Сенсора")
    target_ratio = st.slider("Желаемый Ratio (X:1)", 2.0, 32.0, 10.0, step=1.0)
    exp_long = st.number_input("Выдержка Long (строк)", min_value=2, value=15, step=1)
    
    gain_long = st.slider("Gain Long", 0.5, 5.0, 1.0, step=0.1)
    gain_short = st.slider("Gain Short", 0.1, 5.0, 1.0, step=0.1)

# --- 3. ВЫЧИСЛЕНИЯ ---

# Округляем строки аппаратно (защита от дробных значений)
exp_long_lines = int(exp_long)
exp_short_lines = max(1, int(exp_long_lines / target_ratio))

# Считаем фактический коэффициент склейки
true_ratio = (exp_long_lines * gain_long) / (exp_short_lines * gain_short)

# Генерируем данные
sensor = Sensor10Bit()
scene_lux = generate_fir_tree(max_lux, num_steps=12)

frame_long = sensor.capture(scene_lux, exp_long_lines, gain_long, lens_coeff)
frame_short = sensor.capture(scene_lux, exp_short_lines, gain_short, lens_coeff)

frame_hdr = process_hdr_16bit(frame_long, frame_short, true_ratio)

# --- 4. ВИЗУАЛИЗАЦИЯ ---

st.markdown(f"**Фактические параметры:** Короткая выдержка = `{exp_short_lines}` строк | Итоговый коэффициент склейки = `{true_ratio:.2f}`")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Исходная сцена")
    fig0, ax0 = plt.subplots(figsize=(5, 4))
    ax0.plot(scene_lux, 'orange', label='Освещенность (Lux)')
    ax0.set_yscale('log') # Логарифмическая шкала обязательна для "ёлочки"
    ax0.set_title('Идеальный профиль (Log Scale)')
    ax0.grid(True, which="both", alpha=0.3)
    st.pyplot(fig0)
    
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(frame_long, 'g', label=f'Long ({exp_long_lines} стр.)')
    ax1.plot(frame_short, 'b', label=f'Short ({exp_short_lines} стр.)')
    ax1.axhline(1023, color='r', linestyle=':', label='АЦП Limit (1023)')
    ax1.set_title('Сырые полукадры (10-bit)')
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(frame_hdr, 'm-', linewidth=2, label='HDR Склейка')
    
    # Корректный масштаб для сравнения
    frame_long_scaled = (frame_long / 1023.0) * (65535.0 / true_ratio)
    ax2.plot(frame_long_scaled, 'g--', alpha=0.7, label='LDR (Обрезан)')
    
    ax2.set_title('Результат для конвейера (16-bit)')
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2)

# --- 5. ЭКСПОРТ ДАННЫХ ДЛЯ MODELSIM ---

st.divider()
st.subheader("💾 Экспорт для аппаратного Testbench")
st.markdown("Сохранить текущие массивы в текстовые файлы для загрузки в симулятор (Verilog/C++).")

if st.button("Сгенерировать TXT файлы"):
    # Сохраняем как столбец целых чисел
    np.savetxt("tb_frame_long.txt", frame_long, fmt='%d')
    np.savetxt("tb_frame_short.txt", frame_short, fmt='%d')
    np.savetxt("tb_frame_hdr_ref.txt", frame_hdr, fmt='%d')
    
    st.success("✅ Файлы успешно сохранены: `tb_frame_long.txt`, `tb_frame_short.txt`, `tb_frame_hdr_ref.txt`")