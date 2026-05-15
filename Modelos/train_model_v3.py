import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small

#  Parámetros
IMG_SIZE = 96
BATCH_SIZE = 32

# Generador de datos (SIN augmentación por ahora)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#  Modelo base
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# FASE 1: Congelar TODAS las capas
for layer in base_model.layers:
    layer.trainable = False

# 🔧 Cabeza del modelo
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(7, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Compilar (fase 1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔹 FASE 1: Entrenando solo la cabeza del modelo...\n")

# Entrenamiento FASE 1
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# FASE 2: Descongelar últimas capas
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompilar (fase 2 con LR bajo)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔹 FASE 2: Fine-tuning de las últimas capas...\n")

#  Entrenamiento FASE 2
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Guardar modelo (formato nuevo recomendado)
model.save("modelo_mobilenetv3.keras")

print("\n MobileNetV3 entrenado correctamente en 2 fases")