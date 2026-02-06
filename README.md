# Fashion-MNIST-Image-classification

## google collab link: https://colab.research.google.com/drive/1fNRLGxjYUgCTD0kTlffuIg5wPakigOw8?usp=drive_link

# Questions:
### 1. What is the Fashion MNIST dataset?
  
answer:
The Fashion MNIST dataset contains 70,000 grayscale images of clothing items, which you can found in step 1.3 Training data shape: (60000, 28, 28)
Testing data shape: (10000, 28, 28) which the training images is 60,000  and 10, 000 for test images with the dimension of 28x28 pixels each. It includes 10 difference clothing categories such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot which will be see in step 1.4 class_names = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
] also well see in that step that the images set as a grayscale 
plt.imshow(train_images[0])
plt.colorbar()  # Shows values from 0-255 (grayscale)

### 2. Why do we normalize image pixel values before training?

answer:
We normalize the image pixel values by dividing them by 255 to convert the range from 0-255 to 0-1. We can see that process in step 2.1 Normalize the Data
train_images = train_images / 255.0 test_images = test_images / 255.0 This is done to avoid overflow caused by large numbers, speed up training, and help the neural network learn more efficiently.

### 3. List the layers used in the neural network and their functions.

amswer:
We can see in the step 2.2 
from tensorflow.keras import layers
model = keras.Sequential([
layers.Flatten(input_shape=(28, 28)),
layers.Dense(128, activation='relu'),
layers.Dense(10)
])
model.summary()

output
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten_3 (Flatten)             │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 128)            │       100,480 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 101,770 (397.54 KB)
 Trainable params: 101,770 (397.54 KB)
 Non-trainable params: 0 (0.00 B)

It has three layers the Flatten, Dense(Hidden Layer) and Dense (Output Layer)
first is the flatten layer, the input has dimensions of 28x28 pixel each image which is 2D (rows and columns) but the neural networks need 1D input that why flatten converts 2D to 1D without losing information.

and next is dense layer which has two instances the hidden layer and output layer, first is hidden layer it has 128 number of neurons, it is the hidden layer that connects to flatten layer its 784 values that calculates each neuron (weight × input). Also it is activation relu(Rectified Linear Unit) that introduce non-linearity or allow network to learn complex patterns for example - If value > 0: keep it
- If value ≤ 0: change to 0

Last is the dense ouput layer it has 10 neurons that based on the input vlaue in the dense hidden layer in the step 1.4 class_names = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


### 4. What does an epoch mean in model training?

answer:
An epoch mean ONE complete pass through the ENTIRE training dataset.
we can see that in the  step 2.4 train the model, this code have 10 epoch, then have 1875 batches each epoch, the number of batches results to 60,000 training data shape images divides to 32 the default batch size. Each epoch process 1875 batches 60,000 images that well see in the output that also show the time of process, accuracy and loss percentage

history = model.fit(train_images, train_labels, epochs=10) 

output
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.7809 - loss: 0.6261 #accuracy 78.09% and loss 90.28%
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8626 - loss: 0.3782
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8754 - loss: 0.3413
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8867 - loss: 0.3081
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8945 - loss: 0.2891
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8980 - loss: 0.2788
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9028 - loss: 0.2671

### 5. Compare the predicted label and actual label for the first test image.

answer:
From Step 2.6, the model's predicted label for the first test image is 9, and the actual label is also 9. According to the class names list from Step 1.4, label 9 corresponds to 'Ankle boot'. Since both the predicted and actual labels match, the model correctly classified the first test image as an ankle boot. This demonstrates that the neural network successfully learned to recognize this particular clothing item.

step 2.6
input code
import numpy as np
probability_model = keras.Sequential([
model,
layers.Softmax()
])
predictions = probability_model.predict(test_images)
print("Predicted label for first image:", np.argmax(predictions[0]))
print("Actual label:", test_labels[0])

output 
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 939us/step
Predicted label for first image: 9
Actual label: 9

step 1.4
class_names = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
count it from 0 to 9, the numebr 9 is Ankle boot


### 6. What could be done to improve the model’s accuracy?

answer:
The baseline model achieved 87.56% accuracy. My Task Enhancements showed several improvements:

50 epochs → 88.95% accuracy (best result) - shown in the task enhancement 2
256 neurons → 88.30% accuracy - shown in the task enhancement 1
2 hidden layers → 88.16% accuracy - shown in the task enhancement 3

Additional methods to further improve accuracy include:

Using CNN layers (better for image recognition)
Adding Dropout layers (prevents overfitting)
Implementing data augmentation (more training variety)
Combining multiple enhancements (e.g., 256 neurons + 50 epochs + dropout)
Trying different optimizers or learning rates

