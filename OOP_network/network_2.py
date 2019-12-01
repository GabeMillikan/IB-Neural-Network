import math
import random as _random_module
from networktrainer import double_addition as traindata

_E = 2.71828182845904523536
def random_range(minv = -1, maxv = 1):
    return _random_module.random() * (maxv - minv) + minv

def round(value, nearest = 1):
	return (((value/nearest)+0.5)//1)*nearest
    
def clamp(x, minv, maxv):
    return min(maxv, max(minv, x))
    
def act_sigmoid(x):        
    x = clamp(x, -500, 500) #avoid overflow errors (nearly no change in result)
    return 1/(1+(_E**(-x)))
    
def act_prime_sigmoid(x):
    return act_sigmoid(x)*(1-act_sigmoid(x))
    
def act_ReLU(x):
    return max(0, x)

def act_step(x):
    return 0 if x <=0 else 1

activation = act_sigmoid
activation_prime = act_prime_sigmoid
random = random_range
    
class neuron:
    def __init__(self, node_id, parent_layer):
        self.id = node_id
        self.parent = parent_layer
        
        self.bias = random()
        self.weights = [random() for i in range(self.parent.parent.size[self.parent.id-1])]
        
        self.Ω = random()
        self.α = activation(self.Ω)
        
        self.ε = random() #will be set later
        
    def calculate(self):
        prev_layer = self.parent.parent.layers[self.parent.id-1]
        
        self.Ω = 0
        for i in range(len(prev_layer.neurons)):
            self.Ω += prev_layer.neurons[i].α * self.weights[i]
        
        self.Ω += self.bias
        self.α = activation(self.Ω)
        
class layer:
    def __init__(self, layer_id, parent_network):
        self.id = layer_id
        self.parent = parent_network
        self.neurons = [neuron(i, self) for i in range(self.parent.size[layer_id])]
        
    def calculate(self):
        for n in self.neurons:
            n.calculate()
        
class network:
    predictions_made = 0
    costs_calculated = 0
    cost_average = 0
    
    def __init__(self, dimensions):
        self.size = dimensions
        self.layers = [layer(i, self) for i in range(len(dimensions))]
        
    def feedforward(self, inputs):
        self.predictions_made += 1
        
        for i in range(len(inputs)):
            self.layers[0].neurons[i].α = inputs[i]
            
        for l in range(1, len(self.layers)):
            self.layers[l].calculate()
            
        return [n.α for n in self.layers[-1].neurons]
            
    def cost(self, prediction, actual):
        #prediction is our feedforward prediction
        #actual is the info from a training example
        cost = 0
        for e,a in zip(prediction, actual):
            cost += (e-a)**2
            
        self.cost_average = (self.cost_average*self.costs_calculated + cost)/(self.costs_calculated+1)
        self.costs_calculated+=1
        return cost
        
    def backprop(self, real_input, φ):
        prediction = self.feedforward(real_input)
        
        
        '''
        Example network:
        2 inputs, 2 hidden, 2 outputs
        
        real correlation:
        outputs = innputs, ex: 
                        [1  , 0]   -> [1  , 0]
                        [0  , 1]   -> [0  , 1]
                        [0.5, 0.5] -> [0.5, 0.5]
                        
                        
        
        using sigmoid (σ) as activation. (σ') means sigmoid prime, w.r.t to its only variable
        (x'y) means the derivative of (x) w.r.t (y), or x' if it is a single variable equation
        
        n = neuron
        n[l] = neurons of lth layer (capital L means output layer)
        n[l][k] = neuron of lth layer at position k
        
        w[l][k] = weights of neuron in lth layer at position k
        w[l][k][j] = weight of neuron in lth layer at position k that multiplies with neuron in previous layer at position j
        
        b[l][k] = bias of neuron in lth layer at position k
        
        α = the activation of a neuron
        α[l][k] = σ(Ω[l][k])    (Ω is defined below)
        α[l][k] = the activation of the neuron in lth layer at position k
        
        Ω = the weighted input to a neuron
        Ω[l][k] = (w[l][k][0]*α[l-1][0] + w[l][k][1]*α[l-1][1] + ... + w[l][k][j-1]*α[l-1][j-1] + w[l][k][j]*α[l-1][j]) + b[l][k]
        
        ε = the error in activation of a node, so a high ε means that the node is far from minimizing the cost funciton
        ε[l][k] = error of node in lth layer at position k
        ε is mathamatically defined later
        
        φ = the actual output that was expected based on the input
        φ[k] is the value of the training output at the kth neuron in the output layer
        note: it is not necessary to add a layer index because the expected output (φ) only exists for the output layer (L), so indexing it would be redundant
        
        if α[L] = [1, 0] but φ[L] = [0, 1], then ε[L] is very high, but we need a metric for exactly how bad, that metric is the `cost`
        
        cost = (α[L][0] - φ[0])² + (α[L][1] - φ[1])² + ... + (α[L][k-1] - φ[k-1])² + (α[L][k] - φ[k])²
        cost is the sum of the squares of the errors
        
        in our example, we need to figure out how adjusting each activation will adjust the cost function
        to differentiate (cost'α[L][k]), we do
            = lim h->0: ( ( (α[L][0] - φ[0])² + ... + (α[L][k] + h - φ[k])² ) - ( (α[L][0] - φ[0])² + ... + (α[L][k] - φ[k])² ) )/h
            = lim h->0: ( (α[L][k] + h - φ[k])² - (α[L][k] - φ[k])² ) / h
            (i removed indexing for readability)
            = lim h->0: ( ( (α+h)-φ )² - ( α-φ )² ) /  h
            = lim h->0: ( ( α² + αh - αφ + hα + h² - hφ - φα - φh + φ² ) - ( α² - 2αφ + φ² ) ) / h
            = lim h->0: ( α² + αh - αφ + hα + h² - hφ - φα - φh + φ² - α² + 2αφ - φ² ) / h
            = lim h->0: ( α² - α² + h² + φ² - φ² + αh + hα - αφ - φα + 2αφ - hφ - φh ) / h
            = lim h->0: ( h² + ha + hα - φα - φα + 2αφ - φh - φh ) / h
            = lim h->0: ( h² + 2hα - 2φα + 2αφ - 2φh ) / h
            = lim h->0: ( h² + 2hα - 2φh ) / h
            = lim h->0: ( h(h + 2α - 2φ) ) / h
            = lim h->0: ( h + 2α - 2φ )
            = (0) + 2α - 2φ 
            = 2α - 2φ
            = 2(α - φ)
            (add back indexing)
            = 2(α[L][k] - φ[k])
        
        thus (IMPORTANT):
            (cost'α[L][k]) = 2(α[L][k] - φ[k])
            
        using this, we can use a modified `newtons method` called `gradient descent` to know how to adjust our activation
        for example, if (cost'α[L][k]) is positive, then we must adjust α[L][k] negativly to decrease the cost, which is our ultimate goal
        
        however, since there is no direct way of adjusting the activation, the metric is not particularly useful
        a more useful metric would be how much we need to adjust Ω, the weighted input of the neuron
        we can directly adjust Ω because it is a function of weights and the bias of a neuron.
        
        since α is defined as σ(Ω), and we know (cost'α[L][k]) we can use the chain rule to define:
        (IMPORTANT):
            (cost'Ω[L][k]) = (cost'α[L][k]) * (σ'Ω[L][k]) 
        
        we will call this as the `error` (ε) of the output layer, so:
        (IMPORTANT):
            ε[L][k] = (cost'Ω[L][k])
        
        but clearly we dont only want to change the output layer, so we need a way of defining the ε of neurons in terms of the layer after it
        so: ε[l-1] = f(l)
        this way, we can start at the output using the formulas above, and work backwards from there
        
        
        using the above network:
        lets say that our network output α[L] = [0.88, 0.3] when φ = [0, 1]. note that our network is doing pretty poorley
        since σ(2) ~= 0.88 and σ(-0.85) ~= 0.3, we can assume that Ω[L] = [2, -0.85]
        
        we can calculate ε[L] = [ε[L][0], ε[L][1]]:
        ε[L][0] = (cost'Ω[L][0]) = (cost'α[L][0]) * (σ'Ω[L][0]) = (2(α[L][0] - φ[0])) * (σ'2)     = (2(0.88 - 0)) * (0.105) = 0.1848
        ε[L][0] = (cost'Ω[L][1]) = (cost'α[L][1]) * (σ'Ω[L][1]) = (2(α[L][1] - φ[1])) * (σ'-0.85) = (2(0.3 - 1)) * (0.210)  = -0.294
        so:
        ε[L] = [0.1848, -0.294]
        
        as stated above, these numbers mean that we need to negativly adjust the Ω of output[0], and positively Ω of output[1], which seems correct so far!
        
        now that we know how to adjust the Ω of the output layer, what if we want to change layer L-1?
        
        well, lets say that we adjust Ω[L-1][j] by amount (h), how much will that change each neuron in layer (L)?
        so, in terms of out example matrix, how will neurons in hidden layer affect the output layer?
        in the end, we want to know how a change in the hidden layer will affect the cost
        so, to calculate this change in cost for the top neuron in the hidden layer (we'll call this node q, so q = Ω[1][0])
        
        firstly a change in q will be scaled by the σ function, by a factor of (σ'q)
        then, it will be scaled by the weight of the kth neuron, by a factor of w[L][k][0]
        finally, it will be scaled by how much output neuron affects the cost, a factor of ε[L][k]
        
        so :
            ε[l-1][j] = ∑[k]( (σ'Ω[l-1][j]) * w[l][k][j] * ε[l][k] )
        or (IMPORTANT):
            ε[l][j] = ∑[k]( (σ'Ω[l][j]) * w[l+1][k][j] * ε[l+1][k] )
            
        which can also be said as: 
the sum of ( rate of change in sigmoid w.r.t weighted input times the weight relating the current neuron with the next layer's neuron times that neuron's affect on the cost ) for each neuron in the next layer
        
        using all of the above formulas, we can now figure out how any neuron's Ω will affect the cost ε
        but we can't actually change the Ω, we can only change weights and bias
        
        so lets start with how we can change bias:
            (cost'bias[l][k])
            remember that Ω = weights*previous_layers + bias
            so an adjustment (h) in bias
            means that the new_Ω = weights*previous_layers + (bias + h)
            is the same as saying new_Ω = weights*previous_layers + bias + h
            or new_Ω = Ω + h
            so, the rate of change in Ω with respect to bias is equal to 1
            or (Ω[l][k]'b[l][k]) = 1
            
            by chain rule:
            (cost'b[l][k]) = ε[l][k] * (Ω[l][k]'b[l][k])
            (cost'b[l][k]) = ε[l][k] * 1
        (IMPORTANT):
            (cost'b[l][k]) = ε[l][k]
        
        now, how will the weights affect the cost?
            (cost'weight[l][k][j])
            remember that Ω[l][k] = ∑[j]( w[l][k][j]*α[l-1][j] ) + b[l][k]
            so a change in weight[l][k][j] of size (h) will result in a change h*(Ω[l][k]'w[l][k][j])
            where (Ω[l][k]'w[l][k][j])
                = ( (∑[j]( (w[l][k][j]{+h if j==j})*α[l-1][j]  ) + b[l][k]) - (∑[j]( w[l][k][j]*α[l-1][j] ) + b[l][k]) ) /h
                = lim h->0: ( ( (w[l][k][j]+h)*a[l-1][j] ) - ( w[l][k][j]*a[l-1][j] ) ) /h
                (remove indexing for readiblitiy [α = α[l-1][j], w = w[l][k][j]])
                = lim h->0: ( ( (w+h)*a ) - ( w*a ) ) /h
                = lim h->0: ( ( αw + αh ) - ( aw ) ) /h
                = lim h->0: ( αw + αh - αw ) /h
                = lim h->0: ( αh ) /h
                = α
                (add back indexing)
                = α[l-1][j]
                
            by chain rule:
                (cost'w[l][k][j]) = ε[l][k] * (Ω[l][k]'w[l][k][j])
            (IMPORTANT):
                (cost'w[l][k][j]) = ε[l][k] * α[l-1][j]
                
        SUMMARY:
        
            there are 7 important variables
                Ω    : weighted input of a neuron
                ε    : derivative of cost w.r.t Ω (cost'Ω[l][k])
                φ    : actual, perfect output for a given input
                α    : activation of a neuron (σ)
                w    : weight
                b    : bias
                cost : sum of squared errors between φ and α[L], goal is to minimize this
            
            there are 5 important formulas:
                (cost'α[L][k])    = 2(α[L][k] - φ[k])
                (cost'Ω[L][k])    = (cost'α[L][k]) * (σ'Ω[L][k])
                (cost'Ω[l][k])    = ∑[k]( (σ'Ω[l][j]) * w[l+1][k][j] * ε[l+1][k] )
                (cost'b[l][k])    = ε[l][k]
                (cost'w[l][k][j]) = ε[l][k] * α[l-1][j]
                
        '''
        gradient = [[[0, [0 for w in n.weights]] for n in l.neurons] for l in self.layers]
        
        L = len(self.layers)-1
        for l in range(L, -1, -1):
            K = len(self.layers[l].neurons)
            
            for k in range(K):
                neuron = self.layers[l].neurons[k]
                
                #calculate ε
                if l == L:
                    #this is output layer, calculate ε with 2(α[L][k] - φ[k]) * (σ'Ω[L][k])
                    neuron.ε = 2*(neuron.α - φ[k]) * activation_prime(neuron.Ω)
                else:
                    J = len(self.layers[l+1].neurons)
                    #this is other layer, calculate ε with ∑[k]( (σ'Ω[l][j]) * w[l+1][k][j] * ε[l+1][k] )
                    neuron.ε = 0
                    for j in range(J):
                        neuron.ε += activation_prime(neuron.Ω) * self.layers[l+1].neurons[j].weights[k] * self.layers[l+1].neurons[j].ε
                        
                #calculate b ( = ε)
                gradient[l][k][0] = neuron.ε
                
                #calculate w ( = ε * α[l-1][j])
                for i in range(len(self.layers[l-1].neurons)):
                    gradient[l][k][1][i] = neuron.ε * self.layers[l-1].neurons[i].α
                    
                
                    
        return gradient
        
    def SGD_to_adjustment(self, learning_rate, gradients):
        adjustment = gradients[0] #to get the right shape
        gradient_count = len(gradients)
        for l in range(len(adjustment)):
            for k in range(len(adjustment[l])):
                bias = 0
                for gradient in gradients:
                    bias += gradient[l][k][0]
                bias/=gradient_count
                adjustment[l][k][0] = -bias
                for j in range(len(adjustment[l][k][1])):
                    weight = 0
                    for gradient in gradients:
                        weight += gradient[l][k][1][j]
                    weight/=gradient_count
                    adjustment[l][k][1][j] = -weight
        return adjustment
        
    def make_adjustment(self, adjustment):
        for l in range(len(adjustment)):
            for k in range(len(adjustment[l])):
                self.layers[l].neurons[k].bias += adjustment[l][k][0]
                for j in range(len(adjustment[l][k][1])):
                    self.layers[l].neurons[k].weights[j] += adjustment[l][k][1][j]
            
    def __str__(self):
        r = "\n======== Network After " + str(self.predictions_made) + " Prediction"+("s" if self.predictions_made != 1 else "")+" ========\n"
        r += "Average Cost (all time) : " + str(self.cost_average) + "\n"
        for l in range(len(self.layers)):
            layer = self.layers[l]
            
            if l == 0:
                r += "\n  Inputs:\n"
                max_indent =  math.floor(math.log10(len(layer.neurons)-1 if len(layer.neurons)-1!=0 else 1))
                for k in range(len(layer.neurons)):
                    indent = max_indent - math.floor(math.log10(k if k!= 0 else 1))
                    r += "    ["+str(k)+"] "+" "*indent+": " + str(round(self.layers[l].neurons[k].α, 0.0001))[:6] + "\n"
                    
            elif l == len(self.layers)-1:
                r += "\n  Outputs : " + str(len(layer.neurons)) + " neuron" + ("s" if len(layer.neurons) != 1 else "") + "\n"
                for k in range(len(layer.neurons)):
                    neuron = layer.neurons[k]
                    r += "    neuron["+str(k)+"]:\n"
                    r += "      weighted  : " + str(round(neuron.Ω, 0.001))[:6] + "\n"
                    r += "      activated : " + str(round(neuron.α, 0.0001))[:6] + "\n"
                    r += "      error (ε) : " + str(round(neuron.ε, 0.0001))[:6] + "\n"
                    r += "      internals : \n"
                    max_indent =  math.floor(math.log10(len(neuron.weights)-1 if len(neuron.weights)-1!=0 else 1))
                    for j in range(len(neuron.weights)):
                        weight = neuron.weights[j]
                        indent = max_indent - math.floor(math.log10(j if j!= 0 else 1))
                        r += "        weight[" + str(j) + "] " + indent*" " + ": " + str(round(weight, 0.0001))[:6] + "\n"
                    r += "        bias      " + " "*max_indent + ":  " + str(round(neuron.bias, 0.0001))[:6] + "\n"
            else:
                r += "\n  Hidden Layer [" + str(l) + "] : " + str(len(layer.neurons)) + " neuron" + ("s" if len(layer.neurons) != 1 else "") + "\n"
                for k in range(len(layer.neurons)):
                    neuron = layer.neurons[k]
                    r += "    neuron["+str(k)+"]:\n"
                    r += "      weighted  : " + str(round(neuron.Ω, 0.0001))[:6] + "\n"
                    r += "      activated : " + str(round(neuron.α, 0.0001))[:6] + "\n"
                    r += "      error (ε) : " + str(round(neuron.ε, 0.0001))[:6] + "\n"
                    r += "      internals : \n"
                    max_indent =  math.floor(math.log10(len(neuron.weights)-1 if len(neuron.weights)-1!=0 else 1))
                    for j in range(len(neuron.weights)):
                        weight = neuron.weights[j]
                        indent = max_indent - math.floor(math.log10(j if j!= 0 else 1))
                        r += "        weight[" + str(j) + "] " + indent*" " + ": " + str(round(weight, 0.0001))[:6] + "\n"
                    r += "        bias      " + " "*max_indent + ":  " + str(round(neuron.bias, 0.0001))[:6] + "\n"
        
        r += "\n  Inputs:\n"
        max_indent =  math.floor(math.log10(len(self.layers[0].neurons)-1 if len(self.layers[0].neurons)-1!=0 else 1))
        for k in range(len(self.layers[0].neurons)):
            indent = max_indent - math.floor(math.log10(k if k!= 0 else 1))
            r += "    ["+str(k)+"] "+" "*indent+": " + str(round(self.layers[0].neurons[k].α, 0.0001))[:6] + "\n"
        r += "\n  Final Outputs: \n"
        max_indent =  math.floor(math.log10(len(self.layers[-1].neurons)-1 if len(self.layers[-1].neurons)-1!=0 else 1))
        for k in range(len(self.layers[-1].neurons)):
            indent = max_indent - math.floor(math.log10(k if k!= 0 else 1))
            r += "    ["+str(k)+"] " + " "*indent + ": " + str(round(self.layers[-1].neurons[k].α, 0.0001))[:6] + "\n"
        return r
        

training_inputs  = [[random()] for i in range(101)]
training_outputs = training_inputs

s = (1, 1)
n = network(s)

a = 0
for inputs,outputs in zip(training_inputs, training_outputs):
    a += n.cost(n.feedforward(inputs), outputs)
a/=len(training_inputs)
print("cost: ", a)

for i in range(10):
    gradients = [n.backprop(inputs, outputs) for inputs,outputs in zip(training_inputs[i*10:(i+1)*10], training_outputs[i*10:(i+1)*10])]
    adjustment = n.SGD_to_adjustment(0.01, gradients)
    n.make_adjustment(adjustment)
    print(i)
    
a = 0
for inputs,outputs in zip(training_inputs, training_outputs):
    a += n.cost(n.feedforward(inputs), outputs)
a/=len(training_inputs)
print("cost: ", a)

print(n)






























