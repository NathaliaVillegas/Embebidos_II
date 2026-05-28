import cv2
import numpy as np
import tensorflow as tf
import os
import time

# =========================
# CONFIG
# =========================

url = "http://192.168.26.2:8080/video"
MODEL_PATH = "bebidas_model.keras"

class_names = ["coca", "fanta", "pepsi", "salvietti"]

SAVE_DIR = "retrain_data"

for c in class_names:
    os.makedirs(f"{SAVE_DIR}/{c}", exist_ok=True)

# =========================
# MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# CAMARA
# =========================

cap = cv2.VideoCapture(url)

cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # rotación solo visual
    display = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)

    # predicción usa frame ORIGINAL (NO el rotado)
    inp = preprocess(original)

    pred = model.predict(inp, verbose=0)[0]
    idx = np.argmax(pred)
    label = class_names[idx]
    conf = pred[idx]

    # texto en pantalla
    # mostrar TODAS las probabilidades
    y = 50

    for i, cls in enumerate(class_names):
        text = f"{cls}: {pred[i]:.2f}"

        cv2.putText(display, text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        y += 30

    cv2.imshow("Camara", display)

    key = cv2.waitKey(1) & 0xFF

    # salir
    if key == ord('q'):
        break

    # =========================
    # GUARDADO CORRECTO MANUAL
    # =========================

    elif key == ord('c'):
        cls = "coca"
    elif key == ord('f'):
        cls = "fanta"
    elif key == ord('p'):
        cls = "pepsi"
    elif key == ord('s'):
        cls = "salvietti"
    else:
        cls = None

    if cls is not None:
        filename = f"{SAVE_DIR}/{cls}/img_{int(time.time()*1000)}.jpg"
        cv2.imwrite(filename, original)
        print(f"Guardado en {cls}")

cap.release()
cv2.destroyAllWindows()