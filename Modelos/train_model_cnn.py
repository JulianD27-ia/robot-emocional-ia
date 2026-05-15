import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 📂 Generador de datos
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 CNN básico
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(7, activation='softmax'))

# ⚙️ Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🏋️ Entrenamiento
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 💾 Guardar modelo
model.save("modelo_cnn_basico.h5")

print("✅ CNN básico entrenado y guardado")