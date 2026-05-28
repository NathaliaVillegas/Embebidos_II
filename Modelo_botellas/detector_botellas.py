import cv2
import numpy as np
import tensorflow as tf

# =========================================
# CONFIG
# =========================================

MODEL_PATH = "bebidas_model.keras"

class_names = [
    "coca",
    "fanta",
    "pepsi",
    "salvietti"
]

# =========================================
# MODELO
# =========================================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================================
# CAMARA
# =========================================

url = "http://192.168.26.2:8080/video"
cap = cv2.VideoCapture(url)

cv2.namedWindow(
    "Detector",
    cv2.WINDOW_NORMAL
)

# =========================================
# PREPROCESS
# =========================================

def preprocess(img):

    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# =========================================
# LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # =========================================
    # DETECCION DE OBJETO
    # =========================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_detected = False

    # =========================================
    # BUSCAR CONTORNO GRANDE
    # =========================================

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # ajustar este valor
        if area > 80:

            object_detected = True
            break

    # =========================================
    # SOLO SI HAY OBJETO
    # =========================================

    if object_detected:

        inp = preprocess(frame)

        pred = model.predict(
            inp,
            verbose=0
        )[0]

        idx = np.argmax(pred)

        label = class_names[idx]

        conf = pred[idx]

        # umbral de confianza
        if conf > 0.70:

            text = f"{label} {conf:.2f}"

            cv2.putText(
                display,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    else:

        cv2.putText(
            display,
            "Sin objeto",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    # =========================================
    # MOSTRAR
    # =========================================

    cv2.imshow(
        "Detector",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# =========================================
# LIBERAR
# =========================================

cap.release()
cv2.destroyAllWindows()