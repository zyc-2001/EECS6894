import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Enable mixed precision training to improve efficiency and reduce memory usage
mixed_precision.set_global_policy('mixed_float16')

# Load the CIFAR-10 dataset
dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

# Preprocessing: Resize CIFAR-10's 32x32 images to EfficientNet's input size of 224x224
IMG_SIZE = 32  # EfficientNet default input size
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

BATCH_SIZE = 8  # Reduce batch_size to prevent memory issues
# Load only the first 20,000 training samples and 4,000 test samples of CIFAR-10
train_data = dataset['train'].take(20000).map(preprocess_image).batch(BATCH_SIZE).shuffle(10000).prefetch(tf.data.experimental.AUTOTUNE)
val_data = dataset['test'].take(4000).map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Load the EfficientNetB1 model, excluding the top layer, using pre-trained weights
base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add a custom top layer to adapt the model to the CIFAR-10 dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

# Build the complete model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model checkpoint: save the model after every epoch to prevent loss of progress in case of interruptions
checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True)

# Confirm if a GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Train the model, evaluate using the validation set, and record history data
history = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[checkpoint_cb])

# Plot training and validation accuracy and loss curves
def plot_training_history(history):
    # Plot accuracy curves
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# Call the plot function
plot_training_history(history)
