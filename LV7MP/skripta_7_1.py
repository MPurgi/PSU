import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()

model=keras.Sequential()
model.add(layers.Dense(100, activation="relu", input_shape=(784,)))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# TODO: provedi treniranje mreze pomocu .fit()

history = model.fit(x_train_s, y_train_s,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_loss, train_acc = model.evaluate(x_train_s, y_train_s)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s)
print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
import seaborn as sns
import numpy as np

y_pred_train = np.argmax(model.predict(x_train_s), axis=1)
y_pred_test = np.argmax(model.predict(x_test_s), axis=1)

cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Train Set")

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala


#plt.figure(figsize=(6, 6))
#for i in range(9):
#    plt.subplot(3, 3, i + 1)
#    plt.imshow(x_test[idx], cmap="gray")
#    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_test[idx]}")
#    plt.axis("off")
#plt.show()

wrong = np.where(y_pred_test != y_test)[0]
plt.figure(figsize=(6, 6))
for i in range(9):
    idx = wrong[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_test[idx]}")
    plt.axis("off")
plt.show()
