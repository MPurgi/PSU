import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
print(data)

mpg = data[:, 0]
hp = data[:, 3]
wt = data[:, 5]
cyl = data[:, 1]

wt_scaled = wt * 10

plt.scatter(mpg, hp, s=wt_scaled)

mpg_6_cyl = mpg[cyl == 6]

plt.xlabel("mpg")
plt.ylabel("hp")
plt.title("Ovisnost potrošnje automobila (mpg) o konjskoj snazi (hp)")

print("Najmanja potrošnja automobila:", mpg.min())
print("Najveća potrošnja automobila:", mpg.max())
print("Prosjek potrošnje automobila:", mpg.mean())

print("Automobili sa 6 cilindara imaju prosječnu potrošnju:", mpg_6_cyl.mean())
print("Automobili sa 6 cilindara imaju najmanju potrošnju:", mpg_6_cyl.min())
print("Automobili sa 6 cilindara imaju najveću potrošnju:", mpg_6_cyl.max())

plt.show()