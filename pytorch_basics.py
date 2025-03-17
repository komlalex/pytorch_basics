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
#print("dy/dx", x.grad) 
#print("dy/dw", w.grad)
#print("dy/db", b.grad)


"""TENSOR FUNCTIONS
Apart from arithmetic operations, the torch module also contains many functions 
for creating and manipulating tensors. Let's look at some examples
""" 

# Create a tensor with a fixed value for every element 
t6 = torch.full((3, 2), 42)

# Concatenate two tensors with compatible shapes 
t7 = torch.cat((t3, t6))


# Compute the sin of each element 
t8 = torch.sin(t7)

# Change the shape of a tensor 
t9 = t8.reshape(3, 2, 2)



"""Interoperability with Numpy
Numpy is a popular open-source library used for mathematical and scientific operations
computing in Python. It enables efficient operations on large multi-dimensional
arrays and has a vast ecosystem of supporting libraries, including;
* Pandas for I/O and data analysis
* Matplotlib for plotting and visualization
* OpenCV for image and video processing
"""

# Here's how to create  an array in Numpy 
import numpy as np
x = np.array([[1, 2], [3, 4.]]) 

# We convert a Numpy array to PyTorch tensor using torch.from_numpy
y = torch.from_numpy(x) 
#print(x.dtype, y.dtype)

# We can convert a PyTorch tensor to a numpy array using the .numpy method of a tensor
z = y.numpy() 


"""A LINEAR REGRESSION MODEL FROM SCRATCH"""

"""training data 
We can represent the training data using two matrices: inputs and targets, each with one row per 
observation, and each column per variable
"""
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype="float32") 

# Targets (apples, oranges) 
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype="float32") 

# Convert inputs and targets to tensors 
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets) 

""""LINEAR REGRESSION MODEL FROM SCRATCH 
The weightd and biases can also be represented as matrices, initialized as random
values. The first row of w and the first element of b are used to predict the first 
target variable i.e. yield of apples, and similarly, the second for oranges
"""

# Weights and biases 
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True) 
#print(b)

"""torch.randn creates a tensor with the given shape, with elements picked randomly fron normal 
distribution with mean 0 and standard deviation 1. 
Our model is simply a function that performs a matrix multiplication of the inputs and 
weights and weigths w (transposed) and adds the bias b (replicated for each observation)""" 

#print(inputs @ w.t() + b) 
# We define the model as follows 
def model(x): 
    return x @ w.t() + b   

preds  = model(inputs) 

"""@ represents matrix multiplication in PyTorch, and the .t returns the transpose of a tensor.
The matrix obtained by passing the input data into the model is a set of predictions for the target variables"""

# Compare predictions with targets 
#print(preds) 
#print(targets)

"""You can see a big difference between our model's predictions and the actual targets because we've 
initialized our model with random weights and biases. Obviously, we can't expect a randomly initialized model to work""" 
diff = preds - targets
loss = torch.sum(diff * diff) / diff.numel() 
#print(loss)

"""LOSS FUNCTION 
Before we improve our model, we need to evaluate how well our model is performing. We can compare
 the model's predictions with the actual targets using the following methods
 * Calculate the difference between the two matrices (preds and targets)
 * Square all the elements of the difference matrix to remove negative values
 * Calculate the average of the elements in the resulting matrix.

 The result is single number known as the mean square error (MSE)
"""
def mse(t1, t2): 
    diff = t1 - t2 
    return torch.sum(diff * diff) / diff.numel() 
"""
torch.sum returns the sum of all the elements in a tensor
torch.numel returns the number of elements in a tensors
"""
loss = mse(preds, targets)
print(loss)  

"""
Here's how we can interpret the result: On average, each element in the prediction differs from the 
actual target by the square root of the loss. And that's pretty bad, considering the numbers 
we are trying to predict are themselves in the range 50 - 200. The result is called 
the loss because it indicates how bad the model is at predicting the target values. It 
represents information loss in the model: the lower the loss, the better the model. 
"""

"""COMPUTE GRADIENTS
With PyTorch, we can automatically compute the gradients or derivativess of the loss w.r.t to the weights and biases because
they have require_grad set to True. We'll see how this is useful in just a moment.
"""
# Compute gradients 
loss.backward()

"""The gradients are stored in the .grad property of the respective tensors. Not tht the derivatives of the loss w.r.t the weights matrix 
is itself a matrix with the same dimensions""" 
# Gradients for weigths 
#print(w)
#print(w.grad)

"""Adjust weights and biases to reduce the loss
The loss is a quadratic function of our weights and biases, and our objective is to find
the set of weights where the loss is the lowest. An important insight from calculus is that the 
gradient indicates the rate of change of the loss, i.e., the loss function's slope w.r.t the weights and biases. 
If a gradient is positive: 
* increasing the weight element's value slightly will increase the loss 
* decreasing the weight element's value slightly will decrease the loss

If the gradient is negative:
* increasing the weight element's value slightly will decrease the loss 
* decreasing the weight element's value slighly will increase the loss
""" 
#print(w)
#print(w.grad)
#print(w - w.grad * 1e-5)

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5  

"""We multiply the gradients with a very small number (10^-5) in this case to ensure that we don't modify the 
weights by a very large amount. We want to take a small step in the downhill direction of the gradient, not a giant leap. This number 
is called the learning rate of the algorithm. 

We use torch.no_grad to indicate to PyTorch that we shouldn't track, calculate, or modify 
gradiets while updating the weights and biases""" 

# Let's verify that the loss is actually lower 
preds = model(inputs)
loss = mse(preds, targets) 
#print(loss) 

"""Before we proceed, we reset the gradients to zero by invoking the .zero_grad() method. We need to 
do this because PyTorch accumulates gradients. Otherwise, the next time we invoke .backward on the loss, the new gradients values are added to the existing gradients, which 
may lead to unexpected results""" 

w.grad.zero_()
b.grad.zero_() 
print(w.grad)