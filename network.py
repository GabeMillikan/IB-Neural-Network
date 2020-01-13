'''

    THIS CODE WILL SOON BE UPDATED TO HAVE MORE USEFUL COMMENTS
    AND RE-CODE SOME DUCT TAPE FIXES THAT I MADE EARLIER DUE TO
    THE FACT THAT I WAS REACHING THE DUE DATE FOR THIS PROJECT
    
    im really hoping that i get a chance to fix up this code 
    before someone reads it, its quite dirty - sorry (1/12/20)

'''



'''
    imports
'''
import random as _random_module
from os import system as __SYSCALL
import MNIST_loader
import json
import time

'''
    constants
'''
E = 2.71828

'''
    utils
'''
def cls():
    __SYSCALL("cls")

def random(a=-1,b=1):
    return _random_module.random() * (b-a) + a
    
def sigmoid(x):
    return 1/(1+(E**(-x)))
    
def sigmoid_prime(x):
    r = sigmoid(x)
    return (1-r)*(r)
    
def strpad(s, l):
    s = str(s)
    s = s[:l]
    return s + " " * (0 if len(s) == l else l-len(s))

def componetwise(lam, a1, a2):
    return [lam(a1[i], a2[i]) for i in range(len(a1))]

'''
    hyperparameters
'''
activate = sigmoid
activate_deriv = sigmoid_prime
    
'''
    objects
'''
class neuron:
    def __init__(self, parent, id):
        self.parent = parent
        self.id = id
        
        self.z = 0
        self.a = activate(random(0,1))
        
        self.bias = random()
        self.weights = []
        if self.parent.id != 0:
            for j in range(len(self.parent.parent.layers[self.parent.id-1].neurons)):
                self.weights.append(random())
        else:
            self.weights = [0]
            self.bias = 0
            
        self.error = 0
        
        self.tracked_change = [0, [0 for w in self.weights]]
        self.tracked_count = 0
        
    def calculate(self):
        s = self.bias
        for w, n in zip(self.weights, self.parent.parent.layers[self.parent.id-1].neurons):
            s += w*n.a
        self.z = s
        self.a = activate(self.z)
        
class layer:    
    def __init__(self, parent, height, id):
        self.parent = parent
        self.id = id
        
        self.neurons = [neuron(self, k) for k in range(height)]
    
    def calculate(self):
        for n in self.neurons: n.calculate();
        
class network:
    def rinit(self, dimensions):
        self.size = dimensions
        
        #cannot use list comp because nodes require the layer before their own to init
        self.layers = []
        for h, i in zip(dimensions, range(len(dimensions))):
            self.layers.append(layer(self, h, i))
            
    def __init__(self, dimensions):
        return self.rinit(dimensions)
    
    def dumps(self):
        return json.dumps([[[n.bias, n.weights] for n in l.neurons] for l in self.layers])
    
    def loads(self, s):
        a = json.loads(s)
        
        dimensions = tuple([len(l) for l in a])
        self.rinit(dimensions)
        
        for _l in range(len(a)):
            l = a[_l]
            for _n in range(len(l)):
                ndata = l[_n]
                self.layers[_l].neurons[_n].bias = ndata[0]
                self.layers[_l].neurons[_n].weights = ndata[1]
                
        
    def feedforward(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].neurons[i].a = inputs[i]
            
        for l in self.layers[1:]: l.calculate();
        return [n.a for n in self.layers[-1].neurons]
        
    def backprop_record(self, _in, _out):
        prediction = self.feedforward(_in)
        cst = sum(componetwise(lambda x,y: (x-y)**2, _out, prediction))
        
        # how will a change in z of last layer neuron affect the output?
        # cost = (predict - out)**2
        # dcost/dpredict = 2(predict - out) * dpredict/dz
        #                - remember (for last layer): predict = activation(z)
        #                = 2(predict - out) * act'(z)
        for k in range(len(self.layers[-1].neurons)):
            n = self.layers[-1].neurons[k]
            n.error = 2*(prediction[k]-_out[k])*activate_deriv(n.z)
            #explained later
            n.tracked_change[0] += n.error
            n.tracked_change[1] = componetwise(
                lambda x,y:x+y,
                n.tracked_change[1],
                [self.layers[-2].neurons[j].a * n.error for j in range(len(n.weights))]
                )
            n.tracked_count += 1
        
        # considering the layer before the last layer
        # we know how the next layer's z will affect the cost: nL.error
        # how will my z affect the cost?
        # (m = my layer, n = last layer)
        # find: dcost/dzm
        # = how much i affect next layer * how much next layer affects cost
        # = dzn/dzm * dcost/dzn
        # we know dcost/dzn, it is the error of the next layer
        # zn = act(zm)*wnm + bn
        # dzn/dzm = wnm*act'(zm)
        # dcost/dzm = dzn/dzm * dcost/dzn
        # and since there are multiple neurons in the next layer, we must take the sum of the effect from each one
        for _l in range(len(self.layers)-2): # -2 discludes first and last layer
            #start at second to last layer, and head backwards
            l = -(_l + 2)
            for k in range(len(self.layers[l].neurons)):
                nk = self.layers[l].neurons[k]
                tracked_sum = 0
                for j in range(len(self.layers[l+1].neurons)):
                    nj = self.layers[l+1].neurons[j]
                    w = nj.weights[k]
                    tracked_sum += w*activate_deriv(nk.z)*nj.error
                nk.error = tracked_sum
                # what is dcost/dbias?
                # dz/db * dcost/dz
                # dcost/dz = nk.error
                # z = const + b
                # dz/db = 1
                # dcost/dbias = 1 * nk.error 
                nk.tracked_change[0] += nk.error
                # what is dcost/dwj?
                # dcost/dwj = dz/dwj * dcost/dz
                # dcost/dz = nk.error
                # z = aj*wj + const
                # dz/dwj = aj
                # dcost/dwj = aj * nk.error
                nk.tracked_change[1] = componetwise(
                    lambda x,y:x+y,
                    nk.tracked_change[1],
                    [self.layers[l-1].neurons[j].a * nk.error for j in range(len(nk.weights))]
                    )
                nk.tracked_count += 1
                
    def backprop_update(self, learning_rate):
        for l in self.layers[1:]:
            for n in l.neurons:
                n.bias -= n.tracked_change[0] * learning_rate / n.tracked_count
                for j in range(len(n.weights)):
                    n.weights[j] -= n.tracked_change[1][j] * learning_rate / n.tracked_count
                n.tracked_change[0] = 0
                n.tracked_change[1] = [0 for _ in n.tracked_change[1]]
                n.tracked_count = 0
        
    def __str__(self):
        r = '\n'
        
        r += "Network\n"
        for i in range(len(self.layers)):
            r += "\tLayer " + str(i) + "\n"
            for k in range(len(self.layers[i].neurons)):
                n = self.layers[i].neurons[k]
                
                r += "\t\tNeuron " + strpad(k, 3) + "\n"
                r += "\t\t\tz       : " + strpad(n.z, 7) + "\n"
                r += "\t\t\ta       : " + strpad(n.a, 7) + "\n"
                r += "\t\t\tbias    : " + strpad(n.bias, 7) + " δ " + strpad(n.tracked_change[0]/(n.tracked_count if n.tracked_count != 0 else 1), 7) + "\n"
                r += "\t\t\tweights -\n"
                for j in range(len(n.weights)):
                    r += "\t\t\t        " + str(j) + " : " + strpad(n.weights[j], 7) + " δ " + strpad(n.tracked_change[1][j]/(n.tracked_count if n.tracked_count != 0 else 1), 7) + "\n"
                
        r += "\n"
        
        return r
        
cls()

# TRAINDATA
pri = MNIST_loader.print_image
train_imgs = MNIST_loader.mnist["training_images"]
train_lbls = MNIST_loader.mnist["training_labels"]

# LOAD FROM FILE
load_from_file = (input("Would you like to LOAD this network from file?\n[Y/N]: ").lower() == "y")
if load_from_file:
    filename = input("From what file? (ex. 'network.txt')\nfilename: ")
    directory = input("In what directory? (optional)\ndirectory: ")
    load_from_file = directory + ("\\" if len(directory)>0 else "") + filename
else:
    load_from_file = ""

# PROMPT SAVE TO FILE
if load_from_file=="":
    save_to_file = (input("Would you like to SAVE this network to file?\n[Y/N]: ").lower() == "y")
    if save_to_file:
        filename = input("What should it be saved as? (ex. 'network.txt')\nfilename: ")
        directory = input("In what directory? (optional)\ndirectory: ")
        save_to_file = directory + ("\\" if len(directory)>0 else "") + filename
    else:
        save_to_file = ""
else:
    save_to_file = ""

# NETWORK INIT
n = network((28*28, 50, 100, 500, 1000, 10, 10))

# HYPERPARAMETERS
if load_from_file == "":
    lr = float(input("Learning rate : "))
    epochs = int(input("Data count: "))
    batchsize = int(input("Batch size: "))
    epochs //= batchsize
    
    # BEGIN TRAINING
    for epoch in range(epochs):
        s = epoch * batchsize
        e = s + batchsize
        for t in range(s,e):
            n.backprop_record(train_imgs[t], train_lbls[t])
            n.backprop_update(lr)
        cls()  
        pri(train_imgs[s])
        res = n.feedforward(train_imgs[s])
        for g in range(10):
            r = strpad(g, 2) + " : |" + strpad("=" * round(res[g]*20), 20) + "|" + " - " + strpad(round(res[g]*10000)/100, 4) + "%"
            print(r)
        print("\n" + str(round((epoch+1)*100000/(epochs))/1000) + "% complete")
    if save_to_file != "":
        print("Saving to file '" + str(save_to_file) + "'...")
        with open(save_to_file, "w+") as f:
            f.write(n.dumps())
        print("Done! - exiting...")
else:
    with open(load_from_file, "r") as f:
        n.loads(f.read())
    if __name__ == "__main__":
        cls()
        te = 0
        extt = -1
        correct = 0
        while True:
            #feedforward
            res = n.feedforward(train_imgs[te])
            
            #print
            if te >= extt:
                pri(train_imgs[te])
                r = ''
                for g in range(10):
                    r += strpad(g, 2) + " : |" + strpad("=" * round(res[g]*20), 20) + "|" + " - " + strpad(round(res[g]*10000)/100, 4) + "%\n"
                print(r)
                
            #check if correct
            correct += res.index(max(res)) == train_lbls[te].index(max(train_lbls[te]))
            
            #print correct and progress
            if te >= extt:
                print("Correct so far: " + str(correct*100/(te+1)) + " %")
                print("(" + str(te+1) + "/60000)")
                
                #user input
                q = input("\nHow many examples to test? (default = 1):\nExamples: ")
                q = int(q) if q != "" else 0
                extt = q + te
                extt %= len(train_imgs)
                cls()
            else:
                print(strpad("(" + str(te+1) + "/60000)", 30), end = "\r")
            
            #next exmp
            te += 1
            te %= len(train_imgs)















