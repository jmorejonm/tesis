# Vamos a realizar algunos cálculos estadísticos básicos y generar gráficos para visualizar los resultados proporcionados.
# Para esto, usaremos numpy para los cálculos y matplotlib para la visualización.

import numpy as np
import matplotlib.pyplot as plt

# Datos proporcionados
escenarios = ['low_light', 'medium_light', 'high_light']
avg_time_hog = np.array([0.6393964290618896, 0.6640475988388062, 0.6562575101852417])
avg_time_cnn = np.array([18.23383092880249, 18.35410463809967, 18.05962097644806])
avg_faces_hog = np.array([0.0, 0.5, 0.5])
avg_faces_cnn = np.array([1.0, 1.0, 1.0])

# Calculando diferencias de tiempo de procesamiento entre HOG + Linear SVM y MMOD CNN
time_diff = avg_time_cnn - avg_time_hog

# Creando gráficos
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Gráfico de tiempo de procesamiento
axs[0].bar(escenarios, avg_time_hog, width=0.4, label='HOG + Linear SVM', align='center')
axs[0].bar(escenarios, avg_time_cnn, width=0.4, label='MMOD CNN', align='edge')
axs[0].set_ylabel('Tiempo Promedio de Procesamiento (s)')
axs[0].set_title('Comparación de Tiempo de Procesamiento')
axs[0].legend()

# Gráfico de la diferencia de tiempo de procesamiento
axs[1].plot(escenarios, time_diff, marker='o', linestyle='-', color='r')
axs[1].set_ylabel('Diferencia de Tiempo (s)')
axs[1].set_title('Diferencia en Tiempo de Procesamiento entre HOG + Linear SVM y MMOD CNN')

# Gráfico de precisión de detección
width = 0.35  # ancho de las barras
r1 = np.arange(len(escenarios))
r2 = [x + width for x in r1]

axs[2].bar(r1, avg_faces_hog, color='b', width=width, label='HOG + Linear SVM')
axs[2].bar(r2, avg_faces_cnn, color='g', width=width, label='MMOD CNN')
axs[2].set_ylabel('Promedio de Caras Detectadas')
axs[2].set_title('Comparación de Precisión de Detección')
axs[2].set_xticks([r + width/2 for r in range(len(escenarios))])
axs[2].set_xticklabels(escenarios)
axs[2].legend()

plt.tight_layout()
plt.show()

