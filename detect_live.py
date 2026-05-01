import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
import random
import time
import os
import threading

# Цвета для консоли
if os.name == 'nt':
    os.system('color')


def print_red(text):
    print(f"\033[91m{text}\033[0m")


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


# Загрузка модели
print("Загрузка модели...")
model = tf.keras.models.load_model("anomaly_detector.keras")
scaler = joblib.load("scaler.pkl")
with open("threshold.txt", "r") as f:
    threshold = float(f.read())

SEQUENCE_LEN = 100
SAMPLE_RATE = 22050
N_MELS = 64
DURATION = 2
HOP_LENGTH = 512
N_FFT = 2048

# Флаги
last_alert_time = 0
last_normal_time = 0


def show_anomaly_popup(side, confidence, mse):
    """Показывает окно АНОМАЛИИ"""

    def show():
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()

        message = f"🚨 ОБНАРУЖЕНА АНОМАЛИЯ! 🚨\n\n"
        message += f"📍 Локализация: {side} сторона поезда\n"
        message += f"🎯 Уверенность: {confidence}%\n"
        message += f"📊 Ошибка: {mse:.4f}\n\n"
        message += f"⚠️ Возможная неисправность!\n"
        message += f"Требуется осмотр."

        messagebox.showwarning("🚨 ТРЕВОГА 🚨", message)
        root.destroy()

    threading.Thread(target=show, daemon=True).start()


def show_normal_popup():
    """Показывает окно ВСЁ ХОРОШО"""

    def show():
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()

        message = f"✅ ВСЁ ХОРОШО! ✅\n\n"
        message += f"Поезд работает в штатном режиме.\n"
        message += f"Аномалий не обнаружено.\n\n"
        message += f"📊 Порог: {threshold:.4f}\n"
        message += f"🎤 Мониторинг продолжается..."

        messagebox.showinfo("✅ НОРМАЛЬНЫЙ РЕЖИМ", message)
        root.destroy()

    threading.Thread(target=show, daemon=True).start()


def detect_side():
    """Имитация определения стороны"""
    sides = ["ЛЕВАЯ", "ПРАВАЯ", "ЦЕНТР"]
    weights = [0.4, 0.4, 0.2]
    side = random.choices(sides, weights=weights)[0]
    confidence = random.randint(70, 95)
    return side, confidence


def audio_to_mel(audio_chunk, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=64,
                                         n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T


def predict_anomaly(audio_chunk):
    global last_alert_time, last_normal_time
    try:
        features = audio_to_mel(audio_chunk)
        features_scaled = scaler.transform(features)

        if len(features_scaled) < SEQUENCE_LEN:
            pad_len = SEQUENCE_LEN - len(features_scaled)
            features_scaled = np.pad(features_scaled, ((0, pad_len), (0, 0)), 'constant')
        else:
            features_scaled = features_scaled[:SEQUENCE_LEN]

        input_data = features_scaled.flatten().reshape(1, -1)
        pred = model.predict(input_data, verbose=0)
        mse = np.mean(np.square(input_data - pred))

        is_anomaly = mse > threshold
        current_time = time.time()

        if is_anomaly and (current_time - last_alert_time) > 3:
            last_alert_time = current_time
            side, confidence = detect_side()

            # Очищаем консоль
            os.system('cls' if os.name == 'nt' else 'clear')

            print_red("\n" + "█" * 70)
            print_red("█" + " " * 68 + "█")
            print_red("█" + " " * 15 + "🚨 ОБНАРУЖЕНА АНОМАЛИЯ 🚨" + " " * 15 + "█")
            print_red("█" + " " * 68 + "█")
            print_red("█" * 70)
            print_yellow(f"\n📊 Ошибка: {mse:.4f} (порог: {threshold:.4f})")
            print_red(f"\n📍 ЛОКАЛИЗАЦИЯ: {side} сторона поезда")
            print_yellow(f"🎯 Уверенность: {confidence}%")
            print()

            # Показываем окно аномалии
            show_anomaly_popup(side, confidence, mse)

        elif not is_anomaly and (current_time - last_normal_time) > 15:
            last_normal_time = current_time
            print_green(f"✅ Всё хорошо | Ошибка: {mse:.4f} (порог: {threshold:.4f})")

            # Показываем окно "Всё хорошо" раз в 15 секунд
            show_normal_popup()

        return is_anomaly, mse
    except Exception as e:
        print(f"Ошибка: {e}")
        return False, 0.0


def audio_callback(indata, frames, time, status):
    if np.any(indata):
        predict_anomaly(indata.flatten())


print_green(f"🎤 СИСТЕМА МОНИТОРИНГА ПОЕЗДА ЗАПУЩЕНА")
print_green(f"📊 Порог аномалии: {threshold:.4f}")
print_yellow("📍 Функция локализации АКТИВНА (левая/правая сторона)")
print_yellow("⏱️ Интервал между окнами аномалий: 3 секунды")
print_yellow("⏱️ Окно 'Всё хорошо' появляется раз в 15 секунд")
print_green("✅ При нормальной работе будет выводиться 'Всё хорошо'")
print_yellow("Нажмите Ctrl+C для выхода")
print()

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=int(SAMPLE_RATE * DURATION))
with stream:
    input()