import numpy as np
from urllib import request
import gzip
import pickle
import os

mnist_printer_shades = [" ", "░", "▒", "▓", "█"]
def print_image(i):
    img = "+" + "-"*28*2 + "+\n"
    for y in range(28):
        line = "|"
        for x in range(28):
            v = i[y*28 + x]
            l = mnist_printer_shades[int(round(v*4))]
            line += l*2
        img += line + "|\n"
    img += "+" + "-"*28*2 + "+\n"
    print(img)

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        request.urlretrieve(base_url+name[1], name[1])

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)/255
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
            mnist[name[0]] = [[int(i==x) for i in range(10)] for x in mnist[name[0]]]
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    mnist["training_images"] = mnist["training_images"].tolist()
    mnist["test_images"] = mnist["test_images"].tolist()
    return mnist

mnist = 'loading...'
try:
    mnist = load()
except:
    init()
    mnist = load()

for i in os.listdir():
    for f in filename:
        if i==f[1]:
            os.remove(i)
            