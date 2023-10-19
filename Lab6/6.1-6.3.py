import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load base model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers
x = GlobalAveragePooling2D()(base_model.output)  # Global Average Pooling
x = Dense(1024, activation='relu')(x)  # Layer 1 with 1024 nodes and ReLU activation
x = Dense(1024, activation='relu')(x)  # Layer 2 with 1024 nodes and ReLU activation
x = Dense(512, activation='relu')(x)   # Layer 3 with 512 nodes and ReLU activation
preds = Dense(3, activation='softmax')(x)  # Output layer with 3 nodes and softmax activation

# Assign transfer base model + new layers to model
model = Model(inputs=base_model.input, outputs=preds)

# Assign Trainable layers and freeze layer
for layer in model.layers[:86]:
    layer.trainable = False  # Freeze base model
for layer in model.layers[86:]:
    layer.trainable = True  # Unfreeze new added denses

model.summary()

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

'''
end of 6.1
'''

# Create DataGenerator Object
npic = 5
rotation_range = 40
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
batch_size_train = 16
batch_size_val = 16
seed_value = 50      

datagen = ImageDataGenerator(rotation_range=rotation_range,
                             zoom_range=zoom_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             shear_range=shear_range,
                             horizontal_flip=horizontal_flip,
                             preprocessing_function=preprocess_input,
                             fill_mode='nearest')

# Create Train Image generator
train_generator = datagen.flow_from_directory(
    './Train/',  # This is where you specify the path to the main data folder
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size_train,
    class_mode='categorical',
    seed=seed_value,  # You should define the 'seed_value'
    shuffle=True
)

# Create Validation Image generator
val_generator = datagen.flow_from_directory(
    './Validate/',  # This is where you specify the path to the validation data folder
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size_val,
    class_mode='categorical',
    seed=seed_value,  # You should define the 'seed_val'
    shuffle=True
)

# Get a batch of training data
batch1 = train_generator.next()
Img_train = batch1[0]

# Scale the pixel values from [-1.0, 1.0] to [0.0, 1.0]
Img_train = (Img_train + 1.0) / 2.0

# Get a batch of validation data
batch2 = val_generator.next()
Img_val = batch2[0]

# Scale the pixel values from [-1.0, 1.0] to [0.0, 1.0]
Img_val = (Img_val + 1.0) / 2.0

# Determine the number of rows and columns for subplots
num_rows = 2  # Change this to the desired number of rows
num_cols = 3  # Change this to the desired number of columns

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Save training images
for i in range(min(num_rows * num_cols, Img_train.shape[0])):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.imshow(Img_train[i])

plt.savefig('./lab6/image_train.png')

# Save validation images
for i in range(min(num_rows * num_cols, Img_val.shape[0])):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.imshow(Img_val[i])

plt.savefig('./lab6/image_val.png')

# Create Optimizer
opts = Adam(learning_rate=0.0001)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opts, metrics=['accuracy'])

# Define training Generator Parameters
EP = 100  # Number of iterations (epochs)
step_size_train = train_generator.n // train_generator.batch_size
step_size_val = val_generator.n // val_generator.batch_size

# Check if step_size_train equals step_size_val
# If not, you may need to adjust the batch size to make them equal
if step_size_train != step_size_val:
    print("Warning: step_size_train is not equal to step_size_val. Consider adjusting batch size.")

# Training the model
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train,
                              validation_data=val_generator,
                              validation_steps=step_size_val,
                              epochs=EP,
                              verbose=1)

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
axes[0].plot(history.history['loss'], 'b', label='Train Loss')
axes[0].plot(history.history['val_loss'], 'r--', lw=2, label='Validation Loss')
axes[0].set_title('Loss')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend(loc='upper right')

# Plot Accuracy
axes[1].plot(history.history['accuracy'], 'b', label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], 'r--', lw=2, label='Validation Accuracy')
axes[1].set_title('Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(loc='upper right')

# Display the subplots
plt.tight_layout()
plt.show()

# Save the combined figure
plt.savefig('./lab6/6.2_combined_plots.png')

'''
end of 6.2
'''

# Create an ImageDataGenerator for testing
test_generator = datagen.flow_from_directory(
    './Test/',
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=1
)

# Get class id for y_true
y_true = test_generator.classes

# Predict images according to the test_generator
preds = model.predict_generator(test_generator)
print(preds.shape)
print(preds)

# Get predicted class labels
y_pred = np.argmax(preds, axis=1)
print(test_generator.classes)
print(y_pred)

# Calculate confusion matrix, classification report between y_true and y_pred
confusion = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(report)

'''
end of 6.3
'''