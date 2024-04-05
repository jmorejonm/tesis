# Script de Python para crear un escenario y configuraciones de prueba

# Supongamos que tenemos un conjunto de imágenes para testear los algoritmos HOG + Linear SVM y MMOD CNN de Dlib.

import cv2
import dlib
import numpy as np
import time

# Suponiendo que las rutas a las imágenes están en una lista llamada `image_paths`
image_paths = ["/Users/user/Documents/python/ReconocimientoFacial/images/face_0.jpg", "/Users/user/Documents/python/ReconocimientoFacial/images/jose.jpg"]

# Configuraciones de prueba
scenarios = {
    'low_light': {'brightness': 50, 'contrast': 0.5},
    'medium_light': {'brightness': 100, 'contrast': 1.0},
    'high_light': {'brightness': 150, 'contrast': 1.5}
}

# Función para ajustar brillo y contraste de una imagen
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

# Cargamos los modelos de Dlib y OpenCV (HOG)
hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Función para procesar imágenes en diferentes escenarios
def process_images(image_paths, scenarios):
    results = {}
    for scenario_name, settings in scenarios.items():
        scenario_results = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            adjusted_image = adjust_brightness_contrast(image, settings['brightness'], settings['contrast'])

            # Medición de tiempo para HOG + SVM
            start_time_hog = time.time()
            faces_hog, _ = hog_detector.detectMultiScale(adjusted_image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            time_hog = time.time() - start_time_hog
            
            # Medición de tiempo para MMOD CNN
            start_time_cnn = time.time()
            faces_cnn = cnn_face_detector(adjusted_image)
            time_cnn = time.time() - start_time_cnn
            
            scenario_results.append({'image_path': image_path, 'time_hog': time_hog, 'time_cnn': time_cnn, 'faces_hog': len(faces_hog), 'faces_cnn': len(faces_cnn)})
        results[scenario_name] = scenario_results
    return results

# Ejemplo de uso
results = process_images(image_paths, scenarios)

def compare_metrics(results):
    summary = {}
    for scenario, data in results.items():
        total_time_hog, total_time_cnn = 0, 0
        total_faces_hog, total_faces_cnn = 0, 0
        
        for result in data:
            total_time_hog += result['time_hog']
            total_time_cnn += result['time_cnn']
            total_faces_hog += result['faces_hog']
            total_faces_cnn += result['faces_cnn']
        
        avg_time_hog = total_time_hog / len(data)
        avg_time_cnn = total_time_cnn / len(data)
        avg_faces_hog = total_faces_hog / len(data)
        avg_faces_cnn = total_faces_cnn / len(data)
        
        summary[scenario] = {
            'avg_time_hog': avg_time_hog,
            'avg_time_cnn': avg_time_cnn,
            'avg_faces_hog': avg_faces_hog,
            'avg_faces_cnn': avg_faces_cnn
        }
    
    return summary
# Llamamos a la función compare_metrics con los resultados obtenidos
summary = compare_metrics(results)

# Imprimimos el resumen de las comparaciones de métricas
for scenario, metrics in summary.items():
    print(f"Escenario: {scenario}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print("\n")

# Nota: Este script es un ejemplo básico y no ejecutable directamente sin un entorno adecuado y archivos específicos.
# Además, "mmod_human_face_detector.dat" necesita ser descargado y ubicado correctamente.
