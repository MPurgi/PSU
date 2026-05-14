from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="valid", input_shape=(28,28, 1)),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),

    layers.Conv2D(64, (3, 3), activation="relu", padding="valid"),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# TODO: definiraj callbacks

my_callbacks = [
    callbacks.TensorBoard(log_dir="logs", update_freq=100),
    callbacks.ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

# TODO: provedi treniranje mreze pomocu .fit()

history = model.fit(
    x_train_s,
    y_train_s,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks
)

#TODO: Ucitaj najbolji model

best_model = keras.models.load_model("best_model.h5")

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_pred = best_model.predict(x_train_s)
test_pred = best_model.predict(x_test_s)

train_acc=accuracy_score(np.argmax(y_train_s, axis=1), np.argmax(train_pred, axis=1))
test_acc = accuracy_score(np.argmax(y_test_s, axis=1), np.argmax(test_pred, axis=1))

print("Tocnost na skupu za ucenje: ", train_acc)
print("Tocnost na skupu za testiranje: ", test_acc)

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

cm=confusion_matrix(np.argmax(y_test_s, axis=1), np.argmax(test_pred, axis=1))
print("Matrica zabune (test):")
print(cm)