import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
import matplotlib.pyplot as plt

x,y = mnist.data, mnist.target


def plot(num):
 image = num.reshape(28, 28)
 plt.imshow(image, cmap="binary")
 plt.axis("off")
'''
for i in range(5):
    im = x[i]
    w = plot(im)
    plt.show()
'''
x_train,x_test,y_train, y_test = x[:6000], x[6000:], y[:6000], y[6000:]
yt5 = (y_train =='5')
ytt5 = (y_test == '5')


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state = 42)
sgd.fit(x_train,yt5)
x = sgd.predict([x[0]])
print(x)
