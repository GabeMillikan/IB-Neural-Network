import time

class matrix:
  def __init__(self, data):
    self.data = data;

  def __str__(self):
    printed_string = ''
    printed_string += "[ \n"
    for row_num in range(len(self.data)):
      printed_string+="  "
      printed_string+=str(self.data[row_num])
      if not row_num == len(self.data)-1:
        printed_string += ","
      printed_string+="\n"
    printed_string+="]"
    return printed_string

  def create_from_shape(shape): #not a proper member of an instance
    #shape is in as in math, shape[0] is height and shape[1] is width
    data = []
    for y in range(shape[0]):
        new_r = []
        for x in range(shape[1]):
            new_r.append(0)
        data.append(new_r)
    return matrix(data)
    

  def get_shape(self):
    height = len(self.data)
    width = len(self.data[0])
    return width, height

  def multiply(self, other_matrix):
    my_width, my_height = self.get_shape()
    other_width, other_height = other_matrix.get_shape()
    if not (my_width == other_height):
     raise Exception("Cannot multiply: bad shapes");

    new_width = other_width
    new_height = my_height

    new_data = []
    
    for y in range(new_height):
        new_row = []
        
        for x in range(new_width):
            c = 0
            
            for z in range(my_width):
                c += self.data[y][z] * other_matrix.data[z][x]
                
            new_row.append(c)
        new_data.append(new_row)
                
    new_matrix = matrix(new_data)
    return new_matrix


class NeuralNetwork:
    def __init__(self, shape):
        '''
           - 1A 
        iA - 1B - 2A
        iB - 1C - 2B - oA
        iC - 1D - 2C
           - 1E
           
        where 
            i = input
            o = output
            %d = hidden layer index
        
        '''
    
        weight_data = []
        bias_data = []
        
        layer_count = len(shape)
        if layer_count < 2:
            raise Exception("An input and output layer are required")
        for i in range(layer_count-1):
            weight_data.append(matrix.create_from_shape((shape[i], 1)).data)
            bias_data.append(matrix.create_from_shape((shape[i], 1)).data)
            
        self.weights = matrix(weight_data)
        self.biases = matrix(bias_data)
        
        print(self.weights)
        print(self.biases)


nn = NeuralNetwork((5, 2 ,3))

print('x')
time.sleep(100)