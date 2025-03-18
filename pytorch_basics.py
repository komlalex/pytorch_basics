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
#print(loss)  

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
#print(w.grad) 

"""TRAIN THE MODEL USING GRADIENT DESCENT 

As seen above, we reduce the loss and improve our model using the gradient descent optimization 
algorithm using the following steps: 
1. Generate predictionsd 
2. Calculate the loss 
3. Compute the gradients w.r.t the weights and biases 
4. Adjust the weights by subtracting a small quantity proportional to the gradients
5. Reset the gradients to zero 

Let's implement the above step by step 
""" 
# Generate predictions 
preds = model(inputs) 

# Calcuate the loss 
loss = mse(preds, targets) 

# Compute the gradients 
loss.backward() 

# Adjust the weights & reset the gradients 

with torch.no_grad(): 
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5 
    w.grad.zero_()
    b.grad.zero_() 

# With the new weights and biases, the model should have a lower loss 
preds = model(inputs) 
loss = mse(preds, targets) 
#print("loss 1: ", loss) 

"""Train for multiple epocs 

To reduce the loss further, we can repeat the process of adjusting the weights and biases using the gradients 
multiple times. Each iteration is called an epoch. Let's train the model for 100 epochs
"""

# Train for 100 epochs
for i in range(120): 
    preds = model(inputs) 
    loss = mse(preds, targets) 
    loss.backward() 

    with torch.no_grad():
        w -= w.grad * 1e-5 
        b -= b.grad * 1e-5 
        w.grad.zero_()
        b.grad.zero_() 

"""Once again, let's verify that the loss is now lower: """

# Calculate loss 
preds = model(inputs)
loss = mse(preds, targets) 
#print("loss 2: ", loss)  

"""The loss is significatly lower than its initial value. Let's look at the models 
predictions and compare them with the targets"""
preds = model(inputs) 
#print(preds) 
#print(targets) 

"""LINEAR REGRESSION USING PYTORCH BUILT-INS
We've implemented linear regression & gradient descent model using some basic tensor operations. However, 
since this is a common pattern in deep learning, PyTorch provides several built-in functions and classes 
to make it easy to create and train models with just a few lines of code.

Let's begin by importing the torch.nn package from PyTorch, which contains utility classes for building neural networks.
"""
import torch.nn as nn 

"""As before we represent the inputs and targets as matrices""" 
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], dtype="float32") 

# Targets (apples, oranges) 
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119], 
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100],
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], dtype="float32") 


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets) 

"""We are using 15 training examples to illustrate how to work with large datasets in small batches"""
"""Dataset and DataLoader 
We'll create a TensorDataset, which allows access to rows from inputs and targets
as tuples, and provides standard APIs for working with many different types of datasets in PyTorch
"""
from torch.utils.data import TensorDataset

# Define dtaset 
train_ds = TensorDataset(inputs, targets) 
#print(train_ds[0:3])

"""TensorDataset allows  us to access a small section of the training data using the array notation ([0:3])
in the above code. It returns a tuple with two elements. The first element contains the input varaibles for the selected rows, and the second contains the targets 

We'll also create a DataLoader, which can split the data into batches of a predefined size while training. It also provides other utilities like 
shuffling and random sampling of data"""

from torch.utils.data import DataLoader 

# define data loader 
BATCH_SIZE = 5 
train_dl = DataLoader(train_ds, 
                      batch_size=BATCH_SIZE, 
                      shuffle=True) 
"""We can use the dataloader in a for loop. Let's look at an example""" 

for xb, yb in train_dl:
    #print(xb)
    #print(yb) 
    break  

"""nn.Linear 
Instead of initializing the weights & biases manually, we cans define the model using nn.Linear class from PyTorch, 
which does it automatically 
"""
# Define model 
model = nn.Linear(in_features=3, out_features=2) 
#print(model.weight)
#print(model.bias)

"""PyTorch models also have a helpful .parameters method, which returns a list containing all the weight and bias matrices present in a model. 
For our linear model, we have one weight matrix and one bias matrix""" 

# Parameters
list(model.parameters()) 

"""We can use the model to generate predictions in the same ways as before"""
preds = model(inputs) 

"""Loss Function
Instead of defining a loss function manually, we can use the buit-in loss function mse_loss"""
# Import nn.function 
import torch.nn.functional as F 
loss_fn  = F.mse_loss

"""Let's compute the loss for the current predictions"""
loss = loss_fn(preds, targets)

"""Optimizer 
Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer optim.SGD. SGD is short 
for stochastic gradient descent. The term stochastic indicates the samples are selected in random batches instead instead of as a single group."""

# Define optimizer 
opt = torch.optim.SGD(params=model.parameters(), lr=1e-5) 

"""Note that model.parameters() is passed as an argument to optim.SGD so that the optimizer knows which matrices should be 
modified during the update step. Also, we can specify a learning rate that controls the amount by which the parameters are modified."""

"""Train the model 
We are now ready to train the model. We'll follow the same process to implement gradient descent:
1. Generate predictions 
2. Calculate the loss 
3. Compute the gradients w.r.t the weights and biases 
4. Adjust the weights and biases by subtracting a small quatity proportional to the graduient
5. Reset the gradients to zero 

The only change is that we'll work with batches instead of processing the entire training data in every iteration. Let's define
a utility function fit that trains the model for a given number of epochs
"""

# Uitlity function to train the model 
def fit(num_epochs, model, loss_fn, opt: torch.optim.Optimizer, train_dl): 

    # repeat for a given number of epochs 
    for epoch in range(num_epochs): 

        # Train with batches of data 
        for xb, yb in train_dl: 

            # 1. Generate predictions 
            pred = model(xb) 

            # 2. Calaculate the loss
            loss = loss_fn(pred, yb)

            # 3. Compute the gradient 
            loss.backward() 

            # 4. Update parameters using gradients 
            opt.step() 

            # 5. Reset the gradients to zero 
            opt.zero_grad()
        # Print the progress 
        if (epoch + 1) % 10 == 0: 
            print(f"Epoch: {epoch + 1}/{num_epochs} | Loss: {loss:.4f}") 

fit(num_epochs=100, 
    model=model, 
    loss_fn=loss_fn, 
    opt=opt, 
    train_dl=train_dl) 

"""Let's make predictions with our model""" 
preds = model(inputs) 

# Compare with the targets
#print(preds)
#print(targets)

pred = model(torch.tensor([[75, 63, 44.]]))
print(pred) 

"""Feed Forward Neural Networks"""

model2 = nn.Sequential(
    nn.Linear(3, 3), 
    nn.Sigmoid(), 
    nn.Linear(3, 2)
)

opt = torch.optim.SGD(params=model2.parameters(), lr=1e-2) 

fit(num_epochs=100, 
    model=model2, 
    opt=opt, 
    loss_fn=loss_fn, 
    train_dl=train_dl
    )