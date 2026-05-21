import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("checkpoints/best_model.h5")

img_path = "50kmh.png"

orig_img = image.load_img(img_path)
plt.imshow(orig_img)
plt.title("Ulazna slika")
plt.axis("off")
plt.show()

img = image.load_img(img_path, target_size=(48, 48))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
predicted_class = np.argmax(pred)

print("Predikcija modela – klasa (folder):", predicted_class)
