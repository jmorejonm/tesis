import cv2
import dlib
import logging
import time
import psutil

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo de detección de caras de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el predictor de puntos de referencia faciales de dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Abrir el video de la cámara web
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()  # Iniciar el cronómetro

    # Leer un frame del video
    ret, frame = cap.read()

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = detector(gray)

    # Para cada cara detectada, obtener los puntos de referencia y dibujar puntos críticos y del contorno
    for rect in faces:
        shape = predictor(gray, rect)
        
        # Puntos críticos
        critical_points = [17, 21, 22, 26, 36, 39, 42, 45, 27, 30, 31, 35, 48, 54, 51, 57]
        for i in critical_points:
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Puntos del contorno del rostro
        for i in range(0, 17):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Mostrar el resultado
    cv2.imshow('Face Landmarks', frame)

    # Registrar la eficiencia y recursos consumidos
    elapsed_time = time.time() - start_time
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logging.info(f"Frame processed in {elapsed_time:.2f} seconds. CPU: {cpu_percent}%. Memory: {memory_percent}%.")

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
