'''
how this neural network works without object orientation:
    https://youtu.be/8bNIkfRJZpo?t=10
'''

'''
object orientation hierarchy when dimension = (2,3,2,1)

network
    layer       # input
        neuron
        neuron
    layer       # hidden
        neuron
        neuron
        neuron
    layer       # hidden
        neuron
        neuron
    layer       # output
        neuron
        neuron

'''

import random

def round(value = 0, nearest = 1):
	return (((value/nearest)+0.5)//1)*nearest

class neuron:
    def __init__(self, neuron_id, parent):
        self.id = neuron_id
        self.parent_layer = parent
        #note: self == parent.neurons[neuron_id]
        
        # initalize my values at random <<<<<<<<<<<<<=========== IMPORTANT
        self.value = (random.random() * 2)-1
        
        # bias is a single number that is the offset to a linear equation => y = mx + `b`
        self.bias = (random.random() * 2)-1
        
        # weights are "slope" of the linear equation, but the `mx` is a sum of multiple "mx's"
        # ex: my_value = (weight[0]*x[0] + weight[1]*x[1] + weight[2]*x[2] + ... + weight[n]*x[n]) + bias
        # where `x` is the previous layer's values
        self.weights = []
        
        my_layer_id = self.parent_layer.id
        if my_layer_id == 0:
            #dont calculate weights for input layer
            #because input layer is set arbitrarily
            return
        previous_layer = self.parent_layer.parent_network.layers[my_layer_id-1]
        prev_layer_neuron_count = len(previous_layer.neurons)
        
        #each neuron in the previous layer will be multiplied by its own weight
        #here we create those weights (at random)
        for prev_neuron_id in range(prev_layer_neuron_count):
            self.weights.append((random.random() * 2)-1)
            
    def calculate(self):
        my_layer_id = self.parent_layer.id
        previous_layer = self.parent_layer.parent_network.layers[my_layer_id-1]
        prev_neurons = previous_layer.neurons
        
        # will be added to later, remember:
        # my_value = (weight[0]*x[0] + weight[1]*x[1] + weight[2]*x[2] + ... + weight[n]*x[n]) + bias
        my_new_value = 0
        
        #add the "mx" (weight) term
        #note: len(prev_neurons) == len(self.weights)
        for id in range(len(prev_neurons)):
            prev_neuron = prev_neurons[id]
            relevant_weight = self.weights[id]
            
            my_new_value += prev_neuron.value*relevant_weight
            
        #add the "b" (bias) term
        my_new_value += self.bias
        
        #set new value
        self.value = my_new_value #TODO - activation
        
        #return it because why not (this is entirely unnecessary)
        return my_new_value
        
class layer:
    def __init__(self, layer_id, parent):
        self.id = layer_id
        self.parent_network = parent
        #note: self == network.layers[layer_id]
        
        self.neurons = [] # empty table so we can use `append()`
        
        #get the number of neurons im supposed to have from the network
        network_dimensions = self.parent_network.size
        my_neuron_count = network_dimensions[self.id]
        for neuron_id in range(my_neuron_count):
            
            #create a new neuron with it's id and `self` as parent
            #`self` is parent because the new neuron is "my" child
            new_neuron = neuron(neuron_id, self)
            
            #add neuron to this layer's existing neurons
            self.neurons.append(new_neuron)
            
    def calculate(self):
        #just passes the calculation onto the neurons :)
        
        for neuron in self.neurons:
            neuron.calculate()
        
class network:
    def __init__(self, dimensions):
        self.size = dimensions
        
        self.layers = [] # empty table so we can use `append()`
        
        #number of layers in this network
        layer_count = len(self.size)
        for layer_id in range(layer_count):
            
            #create new layer with its id, and `self` as parent
            #`self` is parent because the new layer is "my" child
            new_layer = layer(layer_id, self)
            
            #add layer to this network's existing layers
            self.layers.append(new_layer)
            
    def loss(self, real, predicted):
        #returns a single number that represents how poorely the newtork predicts the input
        #this function is also normally called `cost` function
        #the method used is `sum of squares` and is the most famous loss function for neural networks
        
        value = 0 #initalize
        
        for i in range(len(real)):
            #we square the difference for two reasons:
            # 1.    we always want the "loss" or "error" to be positive because
            #       if the real value is 5, and we overestimate to 10, we were off by 5 units,
            #       or if we underestimate to 0, we were still off by 5 units
            # 2.    there are multiple ways to get the same loss, we can have a couple of large errors
            #       or we can have lots of little errors. Since the goal is to have the network best predict
            #       the majority of cases, it is much better to have lots of little errors than it is to have
            #       a couple of large errors. Therefore, by squaring the addition, we are penalizing large errors
            #       much more than small ones. Ex: an error of 2 units will add 4 units to our loss, but an error of 10 units
            #       will add 100 units to our loss. An error five times as large will contribute twenty-five as much to the loss
            value += (real[i]-predicted[i])**2
        
        #the unit is arbitrary, but just know that a higher number means a worse network
        return value
    
    def learn(self, real_inputs, real_outputs):
        # TODO - Backpropagation
        pass
        
    def activate(self, x):
        # TODO - explain why this is used
        # this is sigmoid, but feel free to replace with ReLU
        
        #mathamatical constant
        E = 2.71828182845904523536
        
        #avoid overflow errors:
        #note: when x == -10, 0.00005 is returned
        #note: when x ==  10, 0.99995 is returned
        if x < -10:
            x = -10
        elif x > 10:
            x = 10
        
        #sigmoid = https://wikimedia.org/api/rest_v1/media/math/render/svg/faaa0c014ae28ac67db5c49b3f3e8b08415a3f2b
            
        return 1/(1+(E**(-x)))
            
    def calculate(self, inputs):
        #this type of function is normally called "predict", or "feedforward"
        
        #assert will error with the message if the <only> if the condition isn't true
        #ex: this will error if your dimensions are (2,3,1) but you input [5,6,7]
        assert len(inputs) == self.size[0], "Input list must be of length " + str(self.size[0]) + "!"
        
        #firstly, set the inputs to the new layer
        for id in range(len(inputs)):
            # input layer => self.layers[0]
            self.layers[0].neurons[id].value = inputs[id]
            
        #then, calculate all the following layers (in order)
        for layer_id in range(1, len(self.layers)): #start the range at `1` because we dont want to calculate input layer
            self.layers[layer_id].calculate()
            
        #finally, return the output
        #   python lesson:  this entire project could be simplified with something called "list comprehension" 
        #                   it would be less readable to someone who is less familiar with python, so i've avoided using it
        resulting_output = []
        #loop through the output neurons, which will be in the last layer
        for output_neuron in self.layers[-1].neurons: # negative index accesses from right to left, ex: ["spam", "ham", "eggs"][-2] == "ham" 
            resulting_output.append(output_neuron.value)
        
        return resulting_output
    
    def __str__(self):
        #makes it so we can convert the network to a string for printing
        #not important to the functionality of the network itself
        #this function is called when we `print(network)` or `str(network)`
        
        output = '\n'
        output += "network\n"
        
        layer_count = len(self.size)
        for layer_id in range(1, layer_count+1):
            if layer_id == 1:
                output += "\n  input layer\n\n"
            elif layer_id == layer_count:
                output += "\n  output layer\n\n"
            else:
                output += "\n  hidden layer " + str(layer_id-1) + "\n\n" #-1 because of input layer
            
            neuron_count = self.size[layer_id-1]
            for neuron_id in range(neuron_count):
                neuron_value = self.layers[layer_id-1].neurons[neuron_id].value
                output += "    neuron -> " + str(round(neuron_value, 0.0001))[:5] + " \n"
                
        return output
        
    def help(self = 0):
        # prints out all of the stuff in this script
        # useful if you know what you're doing but dont want to read this whole thing
        
        print("=========== Object Orientated Network ===========")
        
        print("    class - network")
        print("attribute -   size")
        print("attribute -   layers")
        print(" function -   calculate(inputs)")
        print("")
        
        print("    class - layer")
        print("attribute -   id")
        print("attribute -   parent_network")
        print("attribute -   neurons")
        print(" function -   calculate()")
        print("")
        
        print("    class - neuron")
        print("attribute -   id")
        print("attribute -   parent_layer")
        print("attribute -   weights")
        print("attribute -   bias")
        print("attribute -   value")
        print(" function -   calculate()")
        
        
        
size = (2,3,2,1)
n = network(size)
print(n) #automatically calls the `__str__` function first
n.calculate([1,2])
print(n)

n.learn([1,1], [2])





