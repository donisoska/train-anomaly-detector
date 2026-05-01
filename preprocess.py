import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

def audio_to_mel(audio_path, sr=22050, n_mels=64, duration=10):
    """Загружает аудио и преобразует его в логарифмическую мел-спектрограмму."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        # Создаем мел-спектрограмму
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                             n_fft=2048, hop_length=512)
        # Переводим в логарифмическую шкалу (так лучше для обучения)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Транспонируем, чтобы форма была (время, n_mels)
        return log_mel.T
    except Exception as e:
        print(f"Ошибка с файлом {audio_path}: {e}")
        return None

train_folder = "data/ToyTrain"  # ← ИСПРАВЛЕНО
X_train = []

print("Обработка обучающих файлов...")
for root, dirs, files in os.walk(train_folder):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            features = audio_to_mel(file_path)
            if features is not None:
                X_train.append(features)

if not X_train:  # ← ИСПРАВЛЕНО (убрана скобка)
    raise Exception(f"Не найдено .wav файлов в папке {train_folder}. Проверьте путь!")

# Объединяем все фрагменты и нормализуем данные
X_train = np.vstack(X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Сохраняем данные и нормализатор
np.save("X_train_scaled.npy", X_train_scaled)
joblib.dump(scaler, "scaler.pkl")
print(f"Данные подготовлены. Форма данных: {X_train_scaled.shape}")