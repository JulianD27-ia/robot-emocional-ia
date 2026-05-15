# Importación de librerías necesarias
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------
# CARGA DEL MODELO BASE
# ---------------------------------------------------------

# Se carga la arquitectura MobileNetV2 preentrenada con ImageNet.
# include_top=False elimina la capa final original del modelo,
# permitiendo agregar capas personalizadas para reconocimiento emocional.

base_model = MobileNetV2(
    input_shape=(48, 48, 3),   # Tamaño de entrada de las imágenes
    include_top=False,         # Excluye la capa de clasificación original
    weights='imagenet'         # Utiliza pesos preentrenados
)

# Se congelan las capas del modelo base para evitar modificar
# los pesos aprendidos durante el entrenamiento inicial.

base_model.trainable = False

# ---------------------------------------------------------
# CAPAS PERSONALIZADAS
# ---------------------------------------------------------

# Se aplana la salida del modelo para convertirla en un vector.
x = tf.keras.layers.Flatten()(base_model.output)

# Capa densa con 128 neuronas y función de activación ReLU.
# Permite aprender características específicas del dataset FER2013.
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Capa de salida con 7 neuronas correspondientes a las emociones:
# angry, disgust, fear, happy, neutral, sad y surprise.
# Softmax permite obtener probabilidades para cada clase.
output = tf.keras.layers.Dense(7, activation='softmax')(x)

# Construcción final del modelo
model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=output
)

# ---------------------------------------------------------
# COMPILACIÓN DEL MODELO
# ---------------------------------------------------------

# Se configura el proceso de entrenamiento:
# - Adam: optimizador utilizado
# - categorical_crossentropy: función de pérdida para clasificación multiclase
# - accuracy: métrica de evaluación

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------------
# PREPROCESAMIENTO Y CARGA DEL DATASET
# ---------------------------------------------------------

# Se normalizan los valores de los píxeles dividiéndolos entre 255.
# validation_split=0.2 reserva el 20% de las imágenes para validación.

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Carga de imágenes para entrenamiento
train_data = datagen.flow_from_directory(
    'dataset/train',           # Carpeta del dataset
    target_size=(48,48),       # Redimensionamiento de imágenes
    batch_size=32,             # Número de imágenes por lote
    class_mode='categorical',  # Clasificación multiclase
    subset='training'          # Subconjunto de entrenamiento
)

# Carga de imágenes para validación
val_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # Subconjunto de validación
)

# ---------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------

# Se entrena el modelo durante 10 épocas utilizando
# los datos de entrenamiento y validación.

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ---------------------------------------------------------
# GUARDADO DEL MODELO
# ---------------------------------------------------------

# El modelo entrenado se almacena en formato HDF5 (.h5)
# para ser utilizado posteriormente en pruebas en tiempo real.

model.save("modelo_emociones.h5")

print("Modelo entrenado y guardado correctamente")