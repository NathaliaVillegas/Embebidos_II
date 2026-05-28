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

URL = "http://192.168.26.2:8080/video"

# =========================================
# CAMARA
# =========================================

cap = cv2.VideoCapture(URL)

# =========================================
# MODELO
# =========================================

model = tf.keras.models.load_model(MODEL_PATH)

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
        print("No se pudo leer frame")
        break

    # =========================================
    # TAMAÑO ORIGINAL
    # =========================================

    H, W = frame.shape[:2]

    # =========================================
    # DISPLAY ROTADO
    # =========================================

    display = cv2.rotate(
        frame,
        cv2.ROTATE_90_CLOCKWISE
    )

    # =========================================
    # GRAY
    # =========================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =========================================
    # BAJAR CONTRASTE
    # =========================================

    gray = cv2.addWeighted(
        gray,
        0.55,
        np.full_like(gray, 135),
        0.45,
        0
    )

    # =========================================
    # BLUR
    # =========================================

    blur = cv2.GaussianBlur(
        gray,
        (11, 11),
        0
    )

    # =========================================
    # THRESHOLD
    # =========================================

    _, thresh = cv2.threshold(
        blur,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # =========================================
    # MORFOLOGÍA
    # =========================================

    kernel = np.ones((7, 7), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =========================================
    # DEBUG
    # =========================================

    debug = cv2.cvtColor(
        thresh,
        cv2.COLOR_GRAY2BGR
    )

    # =========================================
    # CONTORNOS
    # =========================================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ordenar por área
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    detections = []

    used_classes = set()

    # =========================================
    # RECORRER CONTORNOS
    # =========================================

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # =========================================
        # FILTROS
        # =========================================

        if area < 33000:
            continue

        if area > (H * W * 0.70):
            continue

        # =========================================
        # BOUNDING BOX
        # =========================================

        x, y, cw, ch = cv2.boundingRect(cnt)

        # =========================================
        # DIBUJAR EN DEBUG
        # =========================================

        cv2.rectangle(
            debug,
            (x, y),
            (x + cw, y + ch),
            (0, 255, 0),
            2
        )

        # =========================================
        # MOSTRAR ÁREA
        # =========================================

        cv2.putText(
            debug,
            f"{int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # =========================================
        # ROI
        # =========================================

        pad = 20

        roi = frame[
            max(0, y - pad):min(H, y + ch + pad),
            max(0, x - pad):min(W, x + cw + pad)
        ]

        if roi.size == 0:
            continue

        # =========================================
        # PREDICCIÓN
        # =========================================

        pred = model.predict(
            preprocess(roi),
            verbose=0
        )[0]

        idx = np.argmax(pred)

        conf = pred[idx]

        if conf < 0.70:
            continue

        label = class_names[idx]

        # =========================================
        # SOLO 1 POR CLASE
        # =========================================

        if label in used_classes:
            continue

        used_classes.add(label)

        detections.append(
            (label, conf)
        )

    # =========================================
    # MOSTRAR DETECCIONES
    # =========================================

    if len(detections) == 0:

        cv2.putText(
            display,
            "No botellas",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:

        for i, (label, conf) in enumerate(detections):

            cv2.putText(
                display,
                f"{label} {conf:.2f}",
                (20, 50 + i * 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # =========================================
    # ROTAR DEBUG
    # =========================================

    debug = cv2.rotate(
        debug,
        cv2.ROTATE_90_CLOCKWISE
    )

    # =========================================
    # ESCALAR DISPLAY
    # =========================================

    display = cv2.resize(
        display,
        (720, 1280)
    )

    debug = cv2.resize(
        debug,
        (720, 1280)
    )

    # =========================================
    # MOSTRAR
    # =========================================

    #cv2.imshow("Camara",display)

    cv2.imshow(
        "Contornos DEBUG",
        debug
    )

    # =========================================
    # TECLAS
    # =========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# =========================================
# CERRAR
# =========================================

cap.release()

cv2.destroyAllWindows()