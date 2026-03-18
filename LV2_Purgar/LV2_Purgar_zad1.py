import numpy as np
import matplotlib.pyplot as plt

points = np.array([
[1, 1],
[2, 2],
[3, 2],
[3, 1],
[1, 1] 
])

x = points[:, 0]
y = points[:, 1]

plt.plot(x, y, 'g', linewidth=3, marker=".", markersize=20)
plt.axis([0,4, 0, 4])
plt.title("Primjer")
plt.xlabel("x os")
plt.ylabel("y os")
plt.show()