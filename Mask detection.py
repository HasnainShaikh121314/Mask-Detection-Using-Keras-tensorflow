import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/Face_model_new2.h5")

# Class labels
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

# Colors for drawing
colors = {
    "with_mask": (0, 255, 0),              # Green
    "without_mask": (0, 0, 255),           # Red
    "mask_weared_incorrect": (0, 255, 255) # Yellow
}

# Input size expected by the model
input_size = (100, 100)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    
    x1, y1, x2, y2 = 200, 120, 500, 400
    
    fr = frame[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb,input_size)

    face_normalized = face_resized / 255.0
    face_input = np.expand_dims(face_normalized, axis=0)
    preds = model.predict(face_input)
    pred_label = classes[np.argmax(preds)]
    color = colors[pred_label]

        # Draw label and box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, pred_label, (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    print(f"Prediction: {preds}, Label: {pred_label}")

    # Show the frame
    cv2.imshow("Face Mask Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
