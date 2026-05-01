import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Быстрое обучение...")

INPUT_DIM = 6400  # 100 * 64

# Минимум данных для скорости
X_train = np.random.random((500, INPUT_DIM))

# Минимальная модель
model = Sequential([
    Dense(64, activation='relu', input_shape=(INPUT_DIM,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(INPUT_DIM, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

# Всего 3 эпохи для скорости
model.fit(X_train, X_train, epochs=3, batch_size=64, verbose=1)

# Порог
pred = model.predict(X_train)
mse = np.mean(np.square(X_train - pred), axis=1)
threshold = np.percentile(mse, 95)

model.save("anomaly_detector.keras")
with open("threshold.txt", "w") as f:
    f.write(str(threshold))

print(f"✅ Готово! Порог: {threshold:.4f}")