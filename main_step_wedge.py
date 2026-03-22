import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# --- 1. ГЕНЕРАТОРЫ И МОДЕЛИ ---

class StepWedgeGenerator:
    """Генератор симметричной 'ёлочки' (пирамидки)."""
    def generate(self, max_lux, num_steps=10, width_px=1000):
        wedge = np.zeros(width_px)
        step_width = width_px // (num_steps * 2)
        center = width_px // 2
        current_lux = max_lux
        for i in range(num_steps):
            left_start = max(0, center - (i + 1) * step_width)
            left_end = center - i * step_width
            right_start = center + i * step_width
            right_end = min(width_px, center + (i + 1) * step_width)
            wedge[left_start:left_end] = current_lux
            wedge[right_start:right_end] = current_lux
            current_lux /= 2.0 
        return wedge

class Sensor10Bit:
    """Модель 10-битного сенсора с физическими ограничениями."""
    def __init__(self, k_sens=20.0):
        self.max_val = 1023.0 
        self.k_sens = k_sens 
        
    def capture(self, scene_lux, exposure, gain=1.0, lens_coefficient=1.0):
        if not isinstance(exposure, int):
            raise ValueError(f"Ошибка! Экспозиция должна быть целой (int), передано: {exposure}")
        sens_lux = scene_lux / lens_coefficient
        signal = (sens_lux / self.k_sens) * exposure * gain
        return np.floor(np.clip(signal, 0, self.max_val)).astype(int)

def normalize_to_16bit(hdr_float_image, ratio):
    """Линейная нормировка HDR данных в формат uint16 (0-65535)."""
    max_val = 1023.0 * ratio 
    normalized = np.clip(hdr_float_image / max_val, 0, 1)
    return (normalized * 65535).astype(np.uint16)

def reconstruct_hdr(long_frame, short_frame, ratio_long_to_short):
    """HDR склейка и приведение к 16-битному стандарту."""
    threshold = 1000 
    hdr_image = np.zeros_like(long_frame, dtype=float)
    mask_good = long_frame < threshold
    hdr_image[mask_good] = long_frame[mask_good]
    hdr_image[~mask_good] = short_frame[~mask_good] * ratio_long_to_short
    return normalize_to_16bit(hdr_image, ratio_long_to_short)

def compute_ratio(exp_long, exp_short, gain_long, gain_short):
    """Вычисление фактического соотношения чувствительностей."""
    return (exp_long * gain_long) / (exp_short * gain_short)

# --- 2. ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(layout="wide", page_title="HDR Endoscopy Simulator")
st.title("🔬 Виртуальный стенд: Staggered HDR")

with st.sidebar:
    st.header("🗺️ Сцена и Оптика")
    MAX_LUX_VAL = st.slider("Яркость в центре (Lux)", 1000, 300000, 150000, step=5000)
    LENS_COEFFICIENT = st.slider("Ослабление объектива (F-stop)", 1.0, 64.0, 10.0, step=1.0)
    STEPS = st.slider("Количество ступеней ёлочки", 5, 20, 12)

    st.header("📸 Параметры Сенсора")
    TARGET_RATIO = st.slider("Желаемый Ratio (X:1)", 2, 64, 16)
    EXP_LONG = st.number_input("Экспозиция Long (строк)", min_value=2, value=32, step=1)
    
    st.divider()
    GAIN_LONG = st.slider("Усиление (Gain) Long", 1.0, 10.0, 1.0, step=0.1)
    GAIN_SHORT = st.slider("Усиление (Gain) Short", 1.0, 10.0, 1.0, step=0.1)
    
    st.header("🎬 Симуляция")
    run_simulation = st.button("🚀 Запустить динамическую сцену")

# --- 3. ФУНКЦИЯ ОТРИСОВКИ ---

def run_render(current_lux):
    # Расчеты
    exp_long_lines = int(EXP_LONG)
    exp_short_lines = max(1, int(exp_long_lines / TARGET_RATIO))
    total_ratio = compute_ratio(exp_long_lines, exp_short_lines, GAIN_LONG, GAIN_SHORT)

    gen = StepWedgeGenerator()
    sensor = Sensor10Bit()

    scene_lux = gen.generate(current_lux, num_steps=STEPS)
    frame_long = sensor.capture(scene_lux, exp_long_lines, GAIN_LONG, LENS_COEFFICIENT)
    frame_short = sensor.capture(scene_lux, exp_short_lines, GAIN_SHORT, LENS_COEFFICIENT)
    frame_hdr = reconstruct_hdr(frame_long, frame_short, total_ratio)

    # Визуализация
    status_placeholder.info(f"📐 **Состояние:** Яркость = `{int(current_lux)}` Lux | Short = `{exp_short_lines}` строк | Ratio = `{total_ratio:.2f}`")
    
    with col1:
        fig_lux, ax_lux = plt.subplots(figsize=(5, 4))
        ax_lux.plot(scene_lux, color='orange', label='Lux')
        ax_lux.set_yscale('log')
        ax_lux.set_ylim(100, 500000)
        ax_lux.set_title('Профиль Lux (Log)')
        ax_lux.grid(True, which="both", alpha=0.3)
        plot_lux.pyplot(fig_lux)

    with col2:
        fig_adc, ax_adc = plt.subplots(figsize=(5, 4))
        ax_adc.plot(frame_long, 'g', label='Long')
        ax_adc.plot(frame_short, 'b', label='Short')
        ax_adc.axhline(1023, color='r', linestyle=':')
        ax_adc.set_ylim(-50, 1100)
        ax_adc.set_title('Сырые данные (10-bit)')
        ax_adc.grid(True)
        plot_adc.pyplot(fig_adc)

    with col3:
        fig_hdr, ax_hdr = plt.subplots(figsize=(5, 4))
        ax_hdr.plot(frame_hdr, 'm-', linewidth=2, label='HDR')
        ldr_scaled = (frame_long / 1023.0) * (65535.0 / total_ratio)
        ax_hdr.plot(ldr_scaled, 'g--', alpha=0.6)
        ax_hdr.set_ylim(-2000, 70000)
        ax_hdr.set_title('Результат (16-bit)')
        ax_hdr.grid(True)
        plot_hdr.pyplot(fig_hdr)
    
    return frame_long, frame_short, frame_hdr

# --- 4. ЗАПУСК ---

status_placeholder = st.empty()
col1, col2, col3 = st.columns(3)
plot_lux = col1.empty()
plot_adc = col2.empty()
plot_hdr = col3.empty()

if run_simulation:
    # Динамический режим: яркость растет от 5000 до MAX_LUX_VAL
    steps_sim = 30
    lux_values = np.linspace(5000, MAX_LUX_VAL, steps_sim)
    
    for val in lux_values:
        f_long, f_short, f_hdr = run_render(val)
        time.sleep(0.05) # Задержка для плавности анимации [cite: 712]
    st.success("Симуляция завершена")
else:
    # Статический режим
    f_long, f_short, f_hdr = run_render(MAX_LUX_VAL)

# --- 5. ЭКСПОРТ ---
st.divider()
if st.button("💾 Сгенерировать файлы для Testbench"):
    np.savetxt("tb_frame_long.txt", f_long, fmt='%d')
    np.savetxt("tb_frame_short.txt", f_short, fmt='%d')
    np.savetxt("tb_frame_hdr_ref.txt", f_hdr, fmt='%d')
    st.success("Данные последнего кадра сохранены в TXT.")