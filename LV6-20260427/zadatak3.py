import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, dendrogram, linkage
from funkcija_6_1 import generate_data

X = generate_data(500, flagc=1)

Z=linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendogram')
plt.xlabel('Uzorci')
plt.ylabel('Udaljenost')
plt.show()

#Dendrogram prikazuje kako se pojedine točke i skupovi postupno spajaju u sve veće klastere.
#Niže spojene točke predstavljaju sličnije podatke, dok se udaljeniji klasteri spajaju na većoj visini.
#Primjenom argumenta method dobivaju se različiti rezultati jer svaka metoda drugačije definira udaljenost
#između klastera.