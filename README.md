# Fashion-MNIST Image Classification

**Google Colab Link:** [Open in Colab](https://colab.research.google.com/drive/1fNRLGxjYUgCTD0kTlffuIg5wPakigOw8?usp=drive_link)

---

## Questions & Answers

### 1. What is the Fashion MNIST dataset?

The Fashion MNIST dataset contains **70,000 grayscale images** of clothing items:
- **60,000 training images** and **10,000 test images**
- Each image is **28×28 pixels**
- **10 clothing categories:**
`````python
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
` `` ← (no space)

This can be verified in Step 1.3 (data shapes) and Step 1.4 (class names and grayscale visualization using `plt.colorbar()` which shows values from 0–255).

---

### 2. Why do we normalize image pixel values before training?

We normalize pixel values by dividing by 255, converting the range from **0–255** to **0–1**:
````python
# Step 2.1 - Normalize the Data
train_images = train_images / 255.0
test_images = test_images / 255.0
` ``

This is done to:
- **Avoid overflow** caused by large numbers
- **Speed up training**
- Help the neural network **learn more efficiently**

---

### 3. List the layers used in the neural network and their functions.

The model is defined in Step 2.2 as follows:
```python
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
` ``

**Model Summary:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (Flatten)               │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       100,480 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 101,770 (397.54 KB)
` ``

**Layer descriptions:**

| Layer | Description |
|-------|-------------|
| **Flatten** | Converts each 28×28 (2D) image into a 784-element (1D) array so the neural network can process it, without losing any information. |
| **Dense – Hidden Layer** | 128 neurons connected to all 784 inputs. Uses **ReLU** activation (`f(x) = x if x > 0, else 0`) to introduce non-linearity and allow the network to learn complex patterns. |
| **Dense – Output Layer** | 10 neurons (one per clothing class) that output raw scores used to determine the predicted category. |

---

### 4. What does an epoch mean in model training?

An **epoch** is one complete pass through the **entire training dataset**.

In Step 2.4, the model is trained for **10 epochs**:
```python
history = model.fit(train_images, train_labels, epochs=10)
` ``

Each epoch processes **1,875 batches** (60,000 images ÷ 32 default batch size). The training output shows time, accuracy, and loss per epoch:
```
Epoch 1/10  → accuracy: 78.09%  | loss: 0.6261
Epoch 2/10  → accuracy: 86.26%  | loss: 0.3782
Epoch 3/10  → accuracy: 87.54%  | loss: 0.3413
Epoch 4/10  → accuracy: 88.67%  | loss: 0.3081
Epoch 5/10  → accuracy: 89.45%  | loss: 0.2891
Epoch 6/10  → accuracy: 89.80%  | loss: 0.2788
Epoch 7/10  → accuracy: 90.28%  | loss: 0.2671
` ``

As epochs increase, the model improves — accuracy rises and loss decreases.

---

### 5. Compare the predicted label and actual label for the first test image.

From Step 2.6, a `Softmax` layer is added to convert raw scores into probabilities:
```python
probability_model = keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(test_images)

print("Predicted label for first image:", np.argmax(predictions[0]))
print("Actual label:", test_labels[0])
` ``

**Output:**
```
Predicted label for first image: 9
Actual label: 9
` ``

Both the predicted and actual labels are **9**, which corresponds to **'Ankle boot'** (index 9 in `class_names`). The model correctly classified the first test image, demonstrating that it successfully learned to recognize this clothing item.

---

### 6. What could be done to improve the model's accuracy?

The **baseline model** achieved **87.56% accuracy**. The following task enhancements were tested:

| Enhancement | Accuracy |
|-------------|----------|
| Baseline (10 epochs, 128 neurons, 1 hidden layer) | 87.56% |
| 256 neurons (Enhancement 1) | 88.30% |
| 50 epochs (Enhancement 2) | **88.95%** ✅ Best |
| 2 hidden layers (Enhancement 3) | 88.16% |

**Additional methods to further improve accuracy:**
- **CNN layers** – Better suited for image recognition tasks
- **Dropout layers** – Reduces overfitting
- **Data augmentation** – Increases training variety
- **Combined enhancements** – e.g., 256 neurons + 50 epochs + dropout
- **Different optimizers or learning rates** – Fine-tune training dynamics
````

> **Note:** The backtick fences (` ``` `) are shown with a space after them in a few places above just to avoid rendering issues in this chat. When you paste into your actual `README.md` file, make sure all closing code fences are exactly three backticks with no spaces: ` ``` `
