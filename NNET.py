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

first_data = [
  [2, 1, 4],
  [0, 1, 1]
]

second_data = [
  [6, 3, -1, 0],
  [1, 1, 0, 4],
  [-2, 5, 0, 2]
]

first_matrix = matrix(first_data)
second_matrix = matrix(second_data)

resulting_matrix = first_matrix.multiply(second_matrix)

print(str(resulting_matrix))

