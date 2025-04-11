import numpy as np
import matplotlib.pyplot as plt

def knnlr(X_train, y_train, X_test, k=5):
    pred = []
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    for x in X_test:
        dis = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        knn = np.argsort(dis)[:k]

        X_k = X_train[knn]
        y_k = y_train[knn]

        A = np.hstack([X_k, np.ones((X_k.shape[0], 1))])
        coeffs = np.linalg.lstsq(A, y_k)[0]
        k_mean = np.dot(np.hstack([x, 1]), coeffs)
        
        pred.append(k_mean)
    
    return np.array(pred)

def train_test_split(X, y, test_size=0.1):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def r2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

k_values = [1, 5, 25, 125]

data1 = np.load('./data1.npz')
X1_train, X1_test, y1_train, y1_test = train_test_split(data1['X'], data1['y'])
for k in k_values:
    predictions = knnlr(X1_train, y1_train, X1_test, k=k)
    rmse = np.sqrt(np.mean((y1_test - predictions) ** 2))
    print(f'RMSE for k={k}: {rmse}')
    plt.figure(figsize=(12, 6))
    plt.scatter(X1_test, y1_test, c='blue', label='Ground Truth', alpha=0.5)
    plt.scatter(X1_test, predictions, c='red', s=80, marker='X', label=f'KNN Predictions (k={k})')
    plt.title(f'KNN Linear Regression Performance (k={k})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'./knn/data1_{k}.png')

data2 = np.load('./data2.npz')
X2_train, X2_test, y2_train, y2_test = train_test_split(data2['X'], data2['y'])
for k in k_values:
    predictions = knnlr(X2_train, y2_train, X2_test, k=k)
    rmse = np.sqrt(np.mean((y2_test - predictions) ** 2))
    print(f'RMSE for k={k}: {rmse}')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X2_test[:, 0], X2_test[:, 1], y2_test, c='blue', label='Ground Truth', alpha=0.5)
    ax.scatter(X2_test[:, 0], X2_test[:, 1], predictions, c='red', s=80, marker='X', label=f'KNN Predictions (k={k})')
    ax.set_title(f'KNN Linear Regression Performance (k={k})')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend()
    plt.savefig(f'./knn/data2_{k}.png')

k_range = range(1, 126)

r_squared_values_data1 = []
for k in k_range:
    predictions = knnlr(X1_train, y1_train, X1_test, k=k)
    r_squared = r2(y1_test, predictions)
    r_squared_values_data1.append(r_squared)
plt.figure(figsize=(12, 6))
plt.plot(k_range, r_squared_values_data1, marker='o', linestyle='-', color='b')
plt.title('R-squared Error vs. k (data1)')
plt.xlabel('k')
plt.ylabel('R-squared Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('./knn/r_squared_data1.png')

r_squared_values_data2 = []
for k in k_range:
    predictions = knnlr(X2_train, y2_train, X2_test, k=k)
    r_squared = r2(y2_test, predictions)
    r_squared_values_data2.append(r_squared)
plt.figure(figsize=(12, 6))
plt.plot(k_range, r_squared_values_data2, marker='o', linestyle='-', color='b')
plt.title('R-squared Error vs. k (data2)')
plt.xlabel('k')
plt.ylabel('R-squared Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('./knn/r_squared_data2.png')
