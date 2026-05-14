import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'test.png'

# Ucitaj sliku
img_original = mpimg.imread('test.png')  # Zamijeni 'test.png' s putanjom do svoje slike
img = color.rgb2gray(img_original)
img = resize(img, (28, 28))

# Prikazi sliku
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')  
plt.show()

# Pripremi sliku - ulaz u mrezu
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# TODO: ucitaj izgradenu mrezu

model = models.load_model("best_model.h5")

# TODO: napravi predikciju za ucitanu sliku pomocu mreze

pred = model.predict(img)
digit = np.argmax(pred)

# TODO: ispis rezultat u terminal
print("\nPredviđena znamenka: ", digit)
print("Vjerojatnosti po klasama (0-9):")
for i, p in enumerate(pred[0]):
    print(f"{i}: {p*100:6.2f}%")
