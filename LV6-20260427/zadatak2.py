import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from funkcija_6_1 import generate_data

#Generiranje podataka
X = generate_data(500, flagc=1)
#Lista u koju spremamo vrijednosti kriterijske funkcije J
J_values = []
#petlja za K = 1 do 10
for k in range(1, 11):
    #Kreiranje KMeans model s k klustera
    kmeans = KMeans(n_clusters=k, n_init=10)
    #Treniranje modela s podacima
    kmeans.fit(X)
    #Vrijednost kriterijske funkcije J (suma kvadrata udaljenosti)
    J_values.append(kmeans.inertia_)
#Crtanje grafa J ovisno o broju klastera
plt.plot(range(1, 11), J_values, marker='o')
#oznake osi
plt.xlabel('Broj klastera (k)')
plt.ylabel("Kriterijska funkcija (J)")
#naslov grafa
plt.title('Elbow metoda za odabir broja klastera')
#Mreža radi lakšeg čitanja grafa
plt.grid(True)
#Prikaz grafa
plt.show()

#Dobiveni rezultati pokazuju da se vrijednost kriterijske funkcije J
#smanjuje kako raste broj klastera.
#To je očekivano jer s većim brojem klastera centri mogu bolje opisati podatke,
#pa su točke bliže svojim centrima.
#Međutim, smanjenje nije jednako veliko za svaki K.
#Optimalni broj klastera određuje se pomoću elbow metode, gdje tražimo
#točku u kojoj se krivulja lomi i stvara lakat. Odnosno gdje se smanjene pretvara u sporo smanjenje,
#to je točka koja predstavlja najbolji kompromis između broja klastera i kvalitete
#grupiranja, pa se uzima kao optimalan broj klastera. U ovom slučaju je to K=3.