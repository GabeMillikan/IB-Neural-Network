# imports
import math
import random as _random_module

# mathamatical constant, e
_E = 2.71828182845904523536

# helper functions
def random(minv = -1, maxv = 1):
    return _random_module.random() * (maxv - minv) + minv
    
def round(value, nearest = 1):
	return (((value/nearest)+0.5)//1)*nearest
    
def clamp(x, minv, maxv):
    return min(maxv, max(minv, x))
    
# network functions
def sigmoid(x):
    x = clamp(x, -500, 500) #avoid overflow errors (nearly no change in result)
    return 1/(1+(_E**(-x)))
    
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    
#define our activation function
activator       = sigmoid
activator_prime = sigmoid_prime
    
#network classes
from gradient import gradient

class neuron:
    def __init__(self, node_id, parent_layer):
        '''
            setup parenting system
        '''
        # the layer object the contains myself as a child
        self.parent = parent_layer
        # the index of self in the parent.neurons list
        self.id = node_id
        
        '''
            declare the properties of this neuron
        '''
        # commonly called 'z'
        self.weighted_input = 0                          
        # commonly called 'α' (alpha)
        self.activation = 0
        # commonly called 'b'
        self.bias = 0
        # commonly called 'w'
        self.weights = []
        # commonly called 'ε' (epsilon)
        self.error = 0
        
        '''
            define each property
        '''
        # will be set programatically
        self.weighted_input = 0
        # will need to re-set this every time that weighted_input is changed
        self.activation = activator(self.weighted_input)
        # must be initalized at random, or all neurons will be identical, and won't train individually
        self.bias = random()
        # must be initalized at random, or all neurons will be identical, and won't train individually
        # there is one weight for each neuron in the previous neuron
        self.weights = [random() for _ in range(self.parent.parent.size[self.parent.id-1])]
        # derivative of cost w.r.t. self.weighted_input
        # will be set programatically
        self.error = 0
        
    def __str__(self):
        r = "Neuron In Layer ["+str(self.parent.id)+"] At Index ["+str(self.id)+"]: \n"
        r += "  weighted  : " + str(round(self.weighted_input, 0.0001))[:6] + "\n"
        r += "  activated : " + str(round(self.activation, 0.0001))[:6] + "\n"
        r += "  error     : " + str(round(self.error, 0.0001))[:6] + "\n" 
        r += "  bias      : " + str(round(self.bias, 0.0001))[:6] + "\n"
        r += "  weights   : " + str([float(str(round(self.weights[j],0.01))[:(4 if self.weights[j] > 0 else 5)]) for j in range(len(self.weights))])
        return r
        
class layer:
    def __init__(self, layer_id, parent_network):
        '''
            setup parenting system
        '''
        # the network object the contains myself as a child
        self.parent = parent_network
        # the index of self in the parent.layers list
        self.id = layer_id
        
        '''
            declare properties of this layer
        '''
        # a list of neuron objects that are contained in this layer
        self.neurons = []
        
        '''
            define each property
        '''
        # neurons are initalized with their index, and myself
        # the amount of neurons in this layer is the size of the network at this layer
        self.neurons = [neuron(k, self) for k in range(self.parent.size[self.id])]
        
    def calculate(self):
        '''
            for each node (k) in this layer,
            k.weighted_input = Σj ( k.weights[j] * parent.layers[id-1].neurons[j].activation ) + k.bias
        '''
        #calculate the values in each neuron
        for k in range(len(self.neurons)):
            cNeuron = self.neurons[k]
            
            # start it at 0
            cNeuron.weighted_input = 0
            
            # add each weight * their respected activation
            for j in range(len(self.parent.layers[self.id-1].neurons)):
                oNeuron = self.parent.layers[self.id-1].neurons[j]
                
                cNeuron.weighted_input += oNeuron.activation * cNeuron.weights[j]
                
            # add the neuron's bias
            cNeuron.weighted_input += cNeuron.bias
            
            # activate it
            cNeuron.activation = activator(cNeuron.weighted_input)
            
        
    def __str__(self):
        r = "Layer [" + str(self.id) + "] has " + str(len(self.neurons)) + " neuron" + ("s" if len(self.neurons) != 1 else "")
        return r
        
        
class network:
    def __init__(self, size):
        '''
           declare properties of this network 
        '''
        # a size of (3,2,5,1) means 3 inputs, 2 hidden, then 5 hidden, then 1 output
        self.size = (0,0)
        # a list of the layers that are contained in this network, from input layer at index[0] and output at index[-1]
        self.layers = []
        
        '''
            define each property
        '''
        # passed as function parameter
        self.size = size
        #there is one layer for each number in self.size
        self.layers = [layer(l, self) for l in range(len(self.size))]
        
    def predict(self, inputs):
        '''
            -this is known as `feedforward`
            -goal is to figure out what are network thinks the output will be based on inputs
            -will also set all of the intermediate value in the network to reflect this guess
        '''
        
        #set the inputs
        for i in range(len(inputs)):
            self.layers[0].neurons[i].activation = inputs[i]
            
        #starting in the layer after the input, pass the calculate command onto the layer
        for layer in self.layers[1:]:
            layer.calculate()
        
        #retrieve the outputs from the output layer
        return [n.activation for n in self.layers[-1].neurons]
        
    def backprop_single(self, real_input, real_output):
        '''
            -real_input and real_output are a single training example
            -this function will return the gradient for this example
             which will be averaged with other gradients to be used
             for Stochastic Gradient Descent
        '''
        my_prediction = self.predict(real_input)
        
    def __str__(self):
        r = "\n"
        r += "Network: " + str(self.size)
        r += "\n  " + str(len(self.layers)) + " Layers: "
        r += "\n    Input has \t\t\t" + str(len(self.layers[0].neurons)) + " neuron" + ("s" if len(self.layers[0].neurons) != 1 else "")
        for h in range(len(self.layers) - 2):
            r += "\n    Hidded Layer["+str(h+1)+"] has \t" + str(len(self.layers[h+1].neurons)) + " neuron" + ("s" if len(self.layers[h+1].neurons) != 1 else "")
        r += "\n    Output has \t\t\t" + str(len(self.layers[-1].neurons)) + " neuron" + ("s" if len(self.layers[-1].neurons) != 1 else "")
        return r
        





n = network((2,1))

a = gradient((1,2))
b = gradient((1,2))
a += 4
b += 2

a/= b
print(a)
















