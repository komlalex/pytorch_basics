import torch 

"""
TENSORS 
At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, metrix pr n-dimensional array.
Let's create a tensor with a single number.
"""

# Number 
t1 = torch.tensor(4.)
#print(t1) 
"""4. is a shorthand for 4.0. It is used to indicate to Python (and PyTorch) that 
you want to create a floating-point number. We can check this by the .dtype attribute of the tensor"""

#print(t1.dtype) 

"""Let's create a more complex tensors""" 

# Vector 
t2 = torch.tensor([1.0, 2, 3, 4]) 

# Matrix 
t3 = torch.tensor([
    [5., 6], 
    [7, 8], 
    [9, 10]
])

# 3-dimensional array 
t4 = torch.tensor([
    [[11, 12, 13], 
     [13, 14, 15]],

    [[15, 16, 17], 
     [17, 18, 19.]]]) 

#print(t1.shape) 

"""Tensors can have any number of dimensions and different lengths long each dimension. We can 
inspect the length along each dimension using .shape""" 

#print(t4.shape) 

"""NB: It is not possible to create tensors with an improper shape"""
#t5 = torch.tensor([
#    [5.0, 6, 11], 
#    [7, 8], 
#    [9, 10]
#])

"""TENSOR OPERATIONS
We can combine tensors with the usual arithmetic operations. Let's look at an example1

"""
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

print(x, w, b)

# Arithmetic operations 
y = w * x + b 

"""As expected 3 * 4 + 5 = 17. What makes PyTorch unique is that we can 
automatically derive y w.r.t the tensors that have w.r.t requires_grad set to True. 
i.e. w and b. This feature of PyTorch is called autograd(automatic gradient)
To compute the derivatives, we can invoke the .backward method on our result"""

# Compute derivative 
print(y.backward()) 

"""The derivatives of y with respect to the input tensors are stored in the .grad property of the respective tensors"""

# Display gradients
print("dy/dx", x.grad) 
print("dy/dw", w.grad)
print("dy/db", b.grad)
