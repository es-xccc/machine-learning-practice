import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(2025)
w2 = np.random.uniform(size=(2, 1))
w1 = np.random.uniform(size=(2, 2))
b2 = np.random.uniform(size=(1, 1))
b1 = np.random.uniform(size=(1, 2))

learning_rate = 0.1
epochs = 100000
loss_epoch = []

for i in range(epochs):
    # forward pass
    a1 = sigmoid(x @ w1 + b1)
    a2 = sigmoid(a1 @ w2 + b2)

    # loss function
    loss = np.mean((y - a2) ** 2)
    loss_epoch.append(loss)

    # backward pass
    dl_dw2 = a1.T @ (-2 * (y - a2) / y.shape[0] * sigmoid_d(a1 @ w2 + b2)) 
    dl_dw1 = x.T @ (-2 * (y - a2) / y.shape[0] * sigmoid_d(a1 @ w2 + b2) @ w2.T * sigmoid_d(x @ w1 + b1))
    dl_db2 = np.mean(-2 * (y - a2) * sigmoid_d(a1 @ w2 + b2))
    dl_db1 = np.mean(-2 * (y - a2) * sigmoid_d(a1 @ w2 + b2) @ w2.T * sigmoid_d(x @ w1 + b1))

    # gradient descent
    w2 -= learning_rate * dl_dw2
    w1 -= learning_rate * dl_dw1
    b2 -= learning_rate * dl_db2
    b1 -= learning_rate * dl_db1

print(f'final loss (rmse): {np.sqrt(loss_epoch[-1])}')

# prediction
for i in range(len(x)):
    a1 = sigmoid(x[i] @ w1 + b1)
    a2 = sigmoid(a1 @ w2 + b2)
    print(f"x: {x[i]}, y: {a2[0][0]:.4f}")

# loss_epoch.jpg
plt.plot(loss_epoch)
plt.title("Loss (mse) vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (mse)")
plt.grid()
plt.savefig("loss_epoch.jpg")

# xor_heatmap.jpg
resolution = 100
x1 = np.linspace(0, 1, resolution)
x2 = np.linspace(0, 1, resolution)
heatmap = np.zeros((resolution, resolution))

for i in range(resolution):
    for j in range(resolution):
        input_data = np.array([[x1[i], x2[j]]])
        a1 = sigmoid(np.dot(input_data, w1) + b1)
        a2 = sigmoid(np.dot(a1, w2) + b2)
        heatmap[j, i] = a2[0][0]

plt.figure()
plt.imshow(heatmap, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm', aspect='auto')
plt.colorbar(label='Prediction')
plt.title("Heatmap of XOR Predictions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("xor_heatmap.jpg")