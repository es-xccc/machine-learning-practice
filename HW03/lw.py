import numpy as np
import matplotlib.pyplot as plt

def lwlr(X_train, y_train, X_test, tau=0.1):
    pred = []
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    for x in X_test:
        w = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * tau ** 2))
        W = np.diag(w)
        
        A = np.dot(X_train.T, np.dot(W, X_train))
        b = np.dot(X_train.T, np.dot(W, y_train))
        
        w = np.linalg.lstsq(A, b)[0]
        y_pred = np.dot(x, w)
        
        pred.append(y_pred)
    
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

tau_values = [0.01, 0.1, 1, 10]

data1 = np.load('./data1.npz')
X1_train, X1_test, y1_train, y1_test = train_test_split(data1['X'], data1['y'])
for tau in tau_values:
    predictions = lwlr(X1_train, y1_train, X1_test, tau=tau)
    rmse = np.sqrt(np.mean((y1_test - predictions) ** 2))
    print(f'RMSE for tau={tau}: {rmse}')
    plt.figure(figsize=(12, 6))
    plt.scatter(X1_test, y1_test, c='blue', label='Ground Truth', alpha=0.5)
    plt.scatter(X1_test, predictions, c='red', s=80, marker='X', label=f'LWLR Predictions (tau={tau})')
    plt.title(f'LWLR Performance (tau={tau})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    tau_str = str(tau).replace('.', '')
    plt.savefig(f'./lw/data1_{tau_str}.png')

data2 = np.load('./data2.npz')
X2_train, X2_test, y2_train, y2_test = train_test_split(data2['X'], data2['y'])
for tau in tau_values:
    predictions = lwlr(X2_train, y2_train, X2_test, tau=tau)
    rmse = np.sqrt(np.mean((y2_test - predictions) ** 2))
    print(f'RMSE for tau={tau}: {rmse}')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X2_test[:, 0], X2_test[:, 1], y2_test, c='blue', label='Ground Truth', alpha=0.5)
    ax.scatter(X2_test[:, 0], X2_test[:, 1], predictions, c='red', s=80, marker='X', label=f'LWLR Predictions (tau={tau})')
    ax.set_title(f'LWLR Performance (tau={tau})')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend()
    tau_str = str(tau).replace('.', '')
    plt.savefig(f'./lw/data2_{tau_str}.png')

tau_range = np.linspace(0.01, 10, 100)

r_squared_values_data1 = []
for tau in tau_range:
    predictions = lwlr(X1_train, y1_train, X1_test, tau=tau)
    r_squared = r2(y1_test, predictions)
    r_squared_values_data1.append(r_squared)

plt.figure(figsize=(12, 6))
plt.plot(tau_range, r_squared_values_data1, marker='o', linestyle='-', color='b')
plt.title('R-squared Error vs. tau (data1)')
plt.xlabel('tau')
plt.ylabel('R-squared Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('./lw/r_squared_data1.png')

r_squared_values_data2 = []
for tau in tau_range:
    predictions = lwlr(X2_train, y2_train, X2_test, tau=tau)
    r_squared = r2(y2_test, predictions)
    r_squared_values_data2.append(r_squared)

plt.figure(figsize=(12, 6))
plt.plot(tau_range, r_squared_values_data2, marker='o', linestyle='-', color='b')
plt.title('R-squared Error vs. tau (data2)')
plt.xlabel('tau')
plt.ylabel('R-squared Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('./lw/r_squared_data2.png')