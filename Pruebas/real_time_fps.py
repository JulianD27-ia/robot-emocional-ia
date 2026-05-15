import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

#  Cargar modelo (usa el mejor: MobileNetV2)
model = load_model("modelo_emociones.h5")

emociones = ["enojo","disgusto","miedo","alegria","tristeza","sorpresa","neutral"]

# Cámara
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(gray, (48,48))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face = face / 255.0
    face = np.reshape(face, (1,48,48,3))

    #  Predicción
    pred = model.predict(face, verbose=0)
    emocion = emociones[np.argmax(pred)]

    #  FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 🖥 Mostrar
    cv2.putText(frame, f"Emocion: {emocion}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Reconocimiento de emociones", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()