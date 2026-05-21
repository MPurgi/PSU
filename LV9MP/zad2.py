import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import os

train_ds_raw = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

validation_ds_raw = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="validation",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

test_ds_raw = image_dataset_from_directory(
    directory='gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

class_names = test_ds_raw.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.prefetch(AUTOTUNE)
validation_ds = validation_ds_raw.prefetch(AUTOTUNE)
test_ds = test_ds_raw.prefetch(AUTOTUNE)

num_classes = 43

model = models.Sequential()

#32 filtera
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                        input_shape=(48, 48, 3)))
model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

#64 filtera
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

#128 filtera
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = 'checkpoints/best_model.h5'

checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

best_model = tf.keras.models.load_model(checkpoint_path)

test_loss, test_acc = best_model.evaluate(test_ds)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

y_true = []
y_pred = []

for images, labels in test_ds_raw:
    preds = best_model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation='vertical')
plt.title("Matrica zabune – GTSRB")
plt.tight_layout()
plt.show()
