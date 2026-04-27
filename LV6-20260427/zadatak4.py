import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

image = mpimg.imread('example_grayscale.png')

pixels = image.reshape(-1, 1)

K=5

kmeans = KMeans(n_clusters=K, n_init=10)
kmeans.fit(pixels)

compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = compressed_pixels.reshape(image.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_image, cmap='gray')
plt.title(f'Kompresirana slika (K={K})')
plt.axis('off')
plt.show()

#Kada mijenjam broj klastera u KMeans kompresiji slike, primjećujem da slika postaje sve grublja kako je broj klastera manji.
#Za manji broj klastera koristi se vrlo malo nijansi sive pa slika izgleda gubi detalje
#Kako se povećava broj klastera, slika postaje sve sličnija originalu jer koristi više tonova.