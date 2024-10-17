# EfficientNet Training with Caltech256 Dataset

This project demonstrates the process of downloading, preparing, and training various versions of EfficientNet on the **Caltech256** dataset. The process involves setting up the environment, downloading the dataset, training multiple EfficientNet models, and recording their performance.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Download the Caltech256 Dataset](#download-the-caltech256-dataset)
3. [Training EfficientNet Models](#training-efficientnet-models)
4. [Saving and Visualizing Results](#saving-and-visualizing-results)
5. [Results Analysis](#results-analysis)



## Environment Setup


### 1. Sets TPU in GCP

![image](https://github.com/user-attachments/assets/d01ed265-f2fa-47dc-869d-a3f1eb957292)


### 2. Verifying TensorFlow Installation

Before proceeding, verify that TensorFlow is correctly installed in your environment.

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

The expected output should show the installed TensorFlow version (e.g., `2.12.0`).

### 3. Installing Necessary Packages

Ensure all necessary packages like `kaggle`, `tensorflow`, and `pandas` are installed. If not, install them using `pip`.

```bash
pip install tensorflow keras pandas kaggle
```

## Download the Caltech256 Dataset

We will use the **Kaggle** API to download the **Caltech256** dataset.

### 1. Get Your Kaggle API Key

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/account).
2. Scroll down to the "API" section and click **Create New API Token**.
3. A `kaggle.json` file will be downloaded. Place this file in the appropriate directory.

### 2. Move the API Key to the Right Location

```bash
mkdir -p ~/.kaggle
mv ~/path_to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the Dataset

Run the following command to download the Caltech-256 dataset:

```bash
kaggle datasets download -d jessicali9530/caltech256
```

### 4. Extract the Dataset

```bash
unzip caltech256.zip -d ./caltech256
```

## Training EfficientNet Models

### Python Script for Training

Using scripts, call efficientNet and train with datasets. Training.py is a Python script for training multiple EfficientNet models (B0, B1, B3, B5, B7) on the Caltech-256 dataset. The script loads the data set, trains each model, and records the training and validation accuracy and losses for each epoch.

## Saving and Visualizing Results

All the training and validation results are saved in a CSV file named `training_results.csv`.

To visualize the results (accuracy, loss) across different EfficientNet models and epochs, you can read the CSV and plot the data using Python libraries like `matplotlib` or `seaborn`. Visualizing_example.py is an example based on matplotlib


## Results Analysis

### Final Output
![image](https://github.com/user-attachments/assets/be319460-bcc8-42d3-bbc7-497bad948a00)

### 1. Training and Validation Accuracy

The left figure shows the trend of training and validation accuracy across different EfficientNet models as the number of epochs increases:

- **Training Accuracy**: As can be seen, the training accuracy of all models improves with the increase of epochs. The EfficientNetB0 model starts with an accuracy of around 0.72, gradually increasing to about 0.96, while more complex models like EfficientNetB7 show similar improvement, starting slightly lower but eventually reaching near 0.97 accuracy.
  
- **Validation Accuracy**: The validation accuracy for all models starts relatively high and rapidly improves after the first epoch. Most models’ validation accuracy tends to stabilize between 0.85 and 0.88, indicating good performance after training. However, some models, like EfficientNetB0, have a validation accuracy slightly lower than their training accuracy, which suggests slight overfitting.

### 2. Training and Validation Loss

The right figure shows the changes in training and validation loss for different EfficientNet models:

- **Training Loss**: The training loss is generally high during the first epoch, with the initial loss for EfficientNetB0 around 1.5, while other models start with an initial loss of around 1.4. This indicates that the prediction error is quite large at the beginning. As training progresses, the loss rapidly decreases and tends to stabilize, eventually converging between 0.4 and 0.6.

- **Validation Loss**: The validation loss follows a similar trend as the training loss. After the first epoch, the loss drops significantly and then stabilizes. The validation loss for EfficientNetB0 is slightly higher than for the other models. It is worth noting that the validation loss is close to the training loss, indicating good generalization ability of the model, with no significant overfitting.

### 3. Why the initial loss is greater than 1
#### Loss Calculation Code Example
```python
# Calculate loss, which includes softmax cross entropy and L2 regularization.
cross_entropy = tf.losses.softmax_cross_entropy(
    logits=logits,
    onehot_labels=labels,
    label_smoothing=FLAGS.label_smoothing)

# Add weight decay to the loss for non-batch-normalization variables.
loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()
     if 'batch_normalization' not in v.name])
```

The loss function calculation uses Softmax cross-entropy loss combined with L2 regularization. Now we will use these two to answer the question.
---
#### How Softmax Cross-Entropy Works:

- **logits** are the unnormalized outputs of the model.
- **onehot_labels** are the target labels in the one-hot encoding format.
- **label_smoothing** is a hyperparameter for label smoothing, which is used to reduce the model's overconfidence in predictions and prevent the output probabilities from being too concentrated on a single class.
#### The Cross-Entropy Loss Calculation Formula:

$$
\text{Loss} = - \sum_i y_i \log(\hat{y}_i)
$$

Where:

- $y_i$ is the true class label (one-hot encoded).
- $\hat{y}_i$ is the predicted probability computed by Softmax.

When the model’s predictions are inaccurate, particularly when the predicted probability is close to 0, $\log(\hat{y}_i)$ becomes very large (approaching negative infinity), resulting in a large cross-entropy loss. This is why the first **loss** during training can be greater than 1.

#### Key Points:

- **High Initial Loss**: Since the model's predictions are inaccurate during early training (for example, misclassifications, and the output probability distribution deviates from the true distribution), the cross-entropy loss is very large, and it is normal for the loss value to be greater than 1.
- **Loss Decreases with Training**: As training progresses, the model’s predicted probabilities become closer to the true labels, and the cross-entropy loss gradually decreases.
---
#### How L2 Regularization Works:

L2 regularization is a way to prevent overfitting by adding a penalty term based on the square of the parameters. It adds a weight decay term to the loss function:

$$
\text{L2 Loss} = \lambda \sum_i w_i^2
$$

Where:

- $w_i$ are the model's trainable parameters (weights).
- $\lambda$ is the regularization coefficient (i.e., **FLAGS.weight_decay**), which controls the weight of the regularization term.

#### Key Points:

- **Regularization Increases Loss**: The L2 regularization term imposes an additional penalty on the trainable parameters of non-Batch Normalization layers. Therefore, the total **loss** is the sum of the **cross_entropy** and the L2 regularization term. Although this regularization usually accounts for only a small portion of the total loss, it still affects the final loss value.

---


#### Answer: Why the First **loss** is Greater Than 1

- **Characteristics of Cross-Entropy Loss**: Cross-entropy loss is very large when the model predictions are inaccurate, especially in the early stages of training. The predicted probabilities for the target class can be much lower than the true value, so the loss function value will be greater than 1.
- **L2 Regularization**: The L2 regularization term in the loss function increases the overall **loss**, which further explains why the **loss** can be higher than just the **cross_entropy**.

Therefore, it is normal for the initial **loss** to be greater than 1. As training progresses and the model learns the data distribution, the loss value will gradually decrease, converging towards a lower value.
