import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from funkcija_6_1 import generate_data

X = generate_data(500, flagc=5)

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.show()

#Kada nekoliko puta pokrenem kod,
#primjećujem da se raspored klastera može malo promijeniti jer KMeans 
#nasumično inicijalizira početne centre. 
#Kod nekih pokretanja rezultat je isti.

#Kada mijenjam način generiranja podataka (flagc),
#dobivaju se različiti oblici skupova. 
#KMeans dobro radi kada su klasteri jasno odvojeni,
#ali je lošiji kod podataka  nepravilnog oblika poput krugova ili mjeseca.
#Zato što ih on postavlja u centar odnosno kao težište,
#iako možda u tom mjestu ili blizu njega nema podataka.