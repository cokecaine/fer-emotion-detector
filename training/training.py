# Install required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))

# Upload your CSV file (e.g., fer2013.csv)
# uploaded = files.upload()
# csv_filename = list(uploaded.keys())[0]
# print(f'Loaded file: {csv_filename}')

df = pd.read_csv('..\detection\fer2013.csv')
print('Shape:', df.shape)
df.head()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Class distribution
plt.figure(figsize=(10, 4))
df['emotion'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.xticks(range(7), EMOTION_LABELS, rotation=45)
plt.title('Emotion Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print('Usage split:')
print(df['Usage'].value_counts())

# Preview some sample images
fig, axes = plt.subplots(2, 7, figsize=(14, 5))
for emotion_id in range(7):
    samples = df[df['emotion'] == emotion_id].head(2)
    for row, (_, sample) in enumerate(samples.iterrows()):
        pixels = np.array(sample['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        axes[row, emotion_id].imshow(pixels, cmap='gray')
        axes[row, emotion_id].axis('off')
        if row == 0:
            axes[row, emotion_id].set_title(EMOTION_LABELS[emotion_id], fontsize=9)
plt.suptitle('Sample Images per Emotion', fontsize=12)
plt.tight_layout()
plt.show()

IMG_SIZE = 48
NUM_CLASSES = 7

def parse_pixels(df):
    """Convert pixel strings to numpy arrays."""
    X = np.array([row.split() for row in df['pixels']], dtype=np.float32)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize to [0, 1]
    y = keras.utils.to_categorical(df['emotion'].values, NUM_CLASSES)
    return X, y

train_df = df[df['Usage'] == 'Training']
val_df   = df[df['Usage'] == 'PublicTest']
test_df  = df[df['Usage'] == 'PrivateTest']

X_train, y_train = parse_pixels(train_df)
X_val,   y_val   = parse_pixels(val_df)
X_test,  y_test  = parse_pixels(test_df)

print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# Data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name='augmentation')

# Create tf.data datasets for efficient training
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print('Datasets ready!')

def build_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='FER_CNN')

model = build_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_fer_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

EPOCHS = 60

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

test_loss, test_acc = model.evaluate(test_ds)
print(f'\nTest Accuracy: {test_acc:.4f}')
print(f'Test Loss:     {test_loss:.4f}')

# Classification report
y_pred = model.predict(test_ds)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print('\nClassification Report:')
print(classification_report(y_true_labels, y_pred_labels, target_names=EMOTION_LABELS))

# Confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Save final model
model.save('fer_model_final.keras')
print('Model saved!')

# Download the best model
files.download('best_fer_model.keras')

# Predict on a random test sample
idx = np.random.randint(0, len(X_test))
sample = X_test[idx]
true_label = EMOTION_LABELS[np.argmax(y_test[idx])]

pred = model.predict(sample[np.newaxis, ...])[0]
pred_label = EMOTION_LABELS[np.argmax(pred)]
confidence = np.max(pred) * 100

plt.figure(figsize=(4, 4))
plt.imshow(sample.squeeze(), cmap='gray')
plt.title(f'True: {true_label} | Predicted: {pred_label} ({confidence:.1f}%)', fontsize=10)
plt.axis('off')
plt.show()

# Show all class probabilities
plt.figure(figsize=(8, 3))
plt.bar(EMOTION_LABELS, pred * 100, color='steelblue')
plt.ylabel('Probability (%)')
plt.title('Prediction Probabilities')
plt.tight_layout()
plt.show()