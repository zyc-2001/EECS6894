import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5, EfficientNetB7
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd

# Load and preprocess dataset
def load_datasets():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '256_ObjectCategories',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset='training',
        seed=123
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '256_ObjectCategories',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset='validation',
        seed=123
    )

    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))

    return train_dataset, val_dataset

# Train multiple EfficientNet models and record results
def train_multiple_models(models_list, model_names, train_dataset, val_dataset, epochs=5):
    results = []
    for model_class, model_name in zip(models_list, model_names):
        model = model_class(weights=None, input_shape=(224, 224, 3), classes=257)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        
        # Record the accuracy and loss for each epoch
        for epoch in range(epochs):
            results.append({
                'model': model_name,
                'epoch': epoch + 1,
                'accuracy': history.history['accuracy'][epoch],
                'val_accuracy': history.history['val_accuracy'][epoch],
                'loss': history.history['loss'][epoch],
                'val_loss': history.history['val_loss'][epoch]
            })
    return results

# Main
if __name__ == "__main__":
    model_classes = [EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5, EfficientNetB7]
    model_names = ["EfficientNetB0", "EfficientNetB1", "EfficientNetB3", "EfficientNetB5", "EfficientNetB7"]

    train_dataset, val_dataset = load_datasets()

    # Train the models and collect the results
    results = train_multiple_models(model_classes, model_names, train_dataset, val_dataset, epochs=5)

    # Convert the results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("training_results.csv", index=False)
    print("Results saved to training_results.csv")