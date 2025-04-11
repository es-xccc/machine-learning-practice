import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def getData(fileName):
    dataFile = open(fileName, 'r')
    x = []
    y = []
    dataFile.readline() #ignore header
    for line in dataFile:
        d, m = line.split(' ')
        x.append(float(d))
        y.append(float(m))
    dataFile.close()
    return (y, x)


x, y = getData('oddExperiment.txt')

linear_coeffs, linear_residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
linear_fit = np.poly1d(linear_coeffs)
quad_coeffs, quad_residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
quad_fit = np.poly1d(quad_coeffs)

linear_r2 = r2_score(y, linear_fit(x))
quad_r2 = r2_score(y, quad_fit(x))

linear_mse = mean_squared_error(y, linear_fit(x))
quad_mse = mean_squared_error(y, quad_fit(x))

plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, linear_fit(x), 'r-', label=f'Fit of degree 1, LSE = {linear_mse:.5f}')
plt.plot(x, quad_fit(x), 'purple', label=f'Fit of degree 2, LSE = {quad_mse:.5f}')
plt.plot(x, linear_fit(x), 'r-', label=f'Fit of degree 1, R2 = {linear_r2:.5f}')
plt.plot(x, quad_fit(x), 'purple', label=f'Fit of degree 2, R2 = {linear_r2:.5f}')
plt.legend()
plt.title('oddExperiment Data')
plt.legend()
plt.savefig('HW02_4.png')