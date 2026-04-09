import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    return 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - \
           1.1622*np.cos(2*0.6067*x) - 0.9443*np.sin(2*0.6067*x)

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    return y + 0.1*varNoise*np.random.normal(0,1,len(y))

# osnovni podaci
x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:,None]
y_measured = y_measured[:,None]

np.random.seed(12)
ind = np.random.permutation(len(x))
train_i = ind[:35]
test_i = ind[35:]

xtrain_raw = x[train_i]
ytrain = y_measured[train_i]
xtest_raw = x[test_i]
ytest = y_measured[test_i]


# tri stupnja polinoma
degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []
preds = {}

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    xtrain = poly.fit_transform(xtrain_raw)
    xtest = poly.transform(xtest_raw)
    xfull = poly.transform(x)

    model = lm.LinearRegression()
    model.fit(xtrain, ytrain)

    preds[d] = model.predict(xfull)

    MSEtrain.append(mean_squared_error(ytrain, model.predict(xtrain)))
    MSEtest.append(mean_squared_error(ytest, model.predict(xtest)))

print("MSEtrain =", MSEtrain)
print("MSEtest =", MSEtest)

# graf usporedbe modela
plt.figure()
plt.plot(x, y_true, 'k', label='pozadinska funkcija')
for d in degrees:
    plt.plot(x, preds[d], label=f'degree {d}')
plt.legend()
plt.grid()
plt.show()

# što ako ima manje/više uzoraka
for N in [20, 200]:
    xN = np.linspace(1,10,N)
    yN = add_noise(non_func(xN))
    xN = xN[:,None]
    yN = yN[:,None]

    poly = PolynomialFeatures(degree=15)
    xNfull = poly.fit_transform(xN)

    model = lm.LinearRegression()
    model.fit(xNfull, yN)

    plt.figure()
    plt.plot(xN, non_func(xN[:,0]), 'k')
    plt.plot(xN, model.predict(xNfull), 'r')
    plt.title(f"N = {N}")
    plt.grid()
    plt.show()
