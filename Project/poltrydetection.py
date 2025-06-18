import tensorflow as tf
from keras.layers import TFSMLayer
import cv2
import numpy as np

# Load model
model = TFSMLayer(
    r"D:\CapstoneProject\ProjectFiles\dataofprojects\archive\teachable-machine-main\model.savedmodel",
    call_endpoint="serving_default"
)

# Load class labels
class_names = open("labels.txt").read().splitlines()

# Load image
file = r"C:\Users\bisma\Downloads\austra-white.jpg"
img = cv2.imread(file)
if img is None:
    print("❌ Failed to load image.")
    exit()

# Convert BGR to RGB ✅
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# Predict
prediction_dict = model(img_input)
prediction = list(prediction_dict.values())[0]  # Get actual prediction

# If prediction already softmaxed, skip reapplying
probabilities = prediction

# Debug output
print("\n🔍 Prediction Probabilities:")
for i, prob in enumerate(probabilities.numpy()[0]):
    print(f"{i}: {class_names[i]} => {prob:.4f}")

# Get top class
class_index = np.argmax(probabilities)
class_name = class_names[class_index]
print(f"\n✅ Predicted: {class_name}")

# Display result
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV display
cv2.putText(img_bgr, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
