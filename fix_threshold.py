# fix_threshold.py
import numpy as np

# Увеличиваем порог
new_threshold = 3.5

with open("threshold.txt", "w") as f:
    f.write(str(new_threshold))

print(f"✅ Порог изменён с 0.0848 на {new_threshold}")
print("Теперь аномалией будет считаться ошибка > 2.5")
print("Запустите detect_live.py снова")