import numpy as np


SMALLNUMBER = 0.00000001
BIGNUMBER = 100000000

def sigmoid_fn(s):
    return 1/(1+np.exp(-s))


#############################


class Module():
    def __init__(self):
        self.states = [] # contains state variables s_k
        self.states_grad = [] # contains derivatives d loss / d s_k. Note that d Loss/ ds_0 is not needed.
        self.layers = [] #contains network modules
        
    def forward(self,X,y):
        s = X
        self.states = [s]
        for l in self.layers:
            s = l.forward(s)
            self.states.append(s)
        loss = self.loss_fn.forward(s,y)
        self.loss = loss
        
        yhat = s
        return np.mean(loss), yhat
        
    
    def backward(self,X,y):
        sgrad = self.loss_fn.backward(y,self.states[-1])
        self.states_grad = [sgrad]
        for i in range(len(self.states)-2,0,-1):
            sgrad = self.layers[i].backward(sgrad,self.states[i])
            self.states_grad.insert(0,sgrad)
            
            
    def update_params(self, stepsize):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer.num_params > 0:
                layer.update_var(self.states_grad[i],self.states[i], stepsize)
                
    
    
class ReLU(Module):
   def __init__(self):
        self.num_params = 0.
        
   def forward(self,s):
        return np.maximum(0,s)
    
   def backward(self,dLdsn,s):
       dL_ds_in = dLdsn.copy()
       dL_ds_in[s <= 0] = 0
       return dL_ds_in
       
    
    
    
class Sigmoid(Module):
     def __init__(self):
        self.num_params = 0.
        
     def forward(self,s):
        return 1/(1+np.exp(-s))
    
     def backward(self,dLdsn,s):
        sig = self.forward(s)
        dsig = sig * (1 - sig)  
        return dLdsn * dsig
        
    
class Linear(Module):
   def __init__(self, num_outparam,num_inparam, weight_std = 1):
        self.W = np.random.randn(num_outparam,num_inparam)*weight_std
        self.b = np.random.randn(num_outparam)*weight_std
        self.num_params = num_inparam*num_outparam + num_outparam
        
   def forward(self,s):
         self.output = np.dot(self.W, s) + np.outer(self.b, np.ones(s.shape[1]))
         return self.output
    
   def backward(self,dLdsn,s):
        self.dW = np.dot(dLdsn, s.T)  # Shape should be the same as self.W

        # Gradient with respect to biases
        self.db = np.sum(dLdsn, axis=1)  # Sum across samples if batch processing

        # Gradient with respect to input
        grad_input = np.dot(self.W.T, dLdsn)
        return grad_input
    
    
   def update_var(self,dLdsn,s, stepsize):
       dW = 0.
       db = 0.
        
   
       batchsize = s.shape[1]
       dW = np.dot(dLdsn,s.T)
       db = np.sum(dLdsn,axis=1)
        


       self.W = self.W - dW*stepsize
       self.b = self.b - db*stepsize
        
       return self.W,self.b
    
        
         
    
    
    
#####################################


class Model():
    def __init__(self):
        self.layers = []  # Contains network modules
        self.loss_fn = None  # Loss function
        self.states = []  # Initialize states
        self.states_grad = []  # Initialize states_grad

    def forward(self, X, y):
       
        s = X
        self.states = [s]
        for l in self.layers:
            s = l.forward(s)
            self.states.append(s)
        loss = self.loss_fn.forward(y,s)
        self.loss = loss
        
        yhat = s
        return np.sum(loss), yhat
        
    def backward(self, X, y):
        sgrad = self.loss_fn.backward(y,self.states[-1])
        self.states_grad = [sgrad]
        for i in range(len(self.states)-2,0,-1):
            sgrad = self.layers[i].backward(sgrad,self.states[i])
            self.states_grad.insert(0,sgrad)
            
           
#################################################
## binary classification
class Loss():
    pass
    
    
class BCELoss(Loss):
    def __init__(self):
        pass
        
    # f(y,yhat) = -log(max(SMALLNUMBER, sigmoid(y*yhat)))
    def forward(self,y,yhat):
       yhat_clipped = np.clip(yhat, SMALLNUMBER, 1 - SMALLNUMBER)  # Clip yhat to avoid log(0)
       loss = -np.mean(y * np.log(yhat_clipped) + (1 - y) * np.log(1 - yhat_clipped))
       return loss
        
    def backward(self,y,yhat):
        yhat_clipped = np.clip(yhat, SMALLNUMBER, 1 - SMALLNUMBER)  # Clip yhat to avoid division by zero
        grad = (yhat_clipped - y) / (yhat_clipped * (1 - yhat_clipped))
        return grad
    
    
    
    
    

class SimpleReluClassNN(Model):
    def __init__(self, num_layers, hidden_width, num_inparam):
        super().__init__()
        # Create layers
        for i in range(num_layers):
            input_dim = num_inparam if i == 0 else hidden_width
            output_dim = 1 if i == num_layers - 1 else hidden_width

            # Add a fully connected layer
            self.layers.append(Linear(input_dim, output_dim))

            # Add a ReLU layer, except for the output layer
            if i < num_layers - 1:
                self.layers.append(ReLU())
    
############################################

## regression



class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)/2

    def backward(self, y_pred, y_true):
        return  (y_pred - y_true) / y_true.size

 
    
    
class SimpleSigmoidRegressNN(Model):
    def __init__(self, input_size, hidden_size, output_size,weight_std=1):
        super().__init__()
        # Initialize layers
        self.layers.append(Linear(input_size, hidden_size,weight_std))  # First linear layer
        self.layers.append(Sigmoid())  # Sigmoid activation function
        self.layers.append(Linear(hidden_size, output_size,weight_std))  # Output linear layer

        # Assuming the loss function for regression (e.g., Mean Squared Error)
        self.loss_fn = MSELoss()
        
        
        
#############################################

class CrossEntropyLoss(Loss):
     def __init__(self):
        pass
     def cross_entropy_loss(self,y,yhat):
         true_label_idx = np.argmax(y)
   
    # Softmax computation
         exp_pred = np.exp(yhat)
         softmax_pred = exp_pred / np.sum(exp_pred)
    
    # Cross entropy loss for this instance
         loss = -np.log(softmax_pred[true_label_idx])
         return loss
       
     def forward(self, y, yhat):
        self.y_pred = yhat  # Store y_pred
        losses = np.array([self.cross_entropy_loss(true, pred) for true, pred in zip(y, yhat)])
        loss_matrix = losses.reshape(-1, 1)
        return loss_matrix
     def backward(self, y,yhat):
        gradients = []
        for true, pred in zip(y, self.y_pred):
            exp_pred = np.exp(pred)
            softmax_pred = exp_pred / np.sum(exp_pred)

            # Compute gradient for this instance
            gradient = softmax_pred - true
            gradients.append(gradient)
        
        return np.array(gradients)
class SimpleReluMulticlassNN(Model):
    def __init__(self, input_size, hidden_size, output_size,weight_std=1):
        super().__init__()
        # Initialize layers
        self.layers.append(Linear(input_size, hidden_size,weight_std))  # First linear layer
        self.layers.append(ReLU())  # ReLU activation function
        self.layers.append(Linear(hidden_size, output_size,weight_std))  # Output linear layer

        # Assuming the loss function for multiclass classification (e.g., Cross-Entropy)
        self.loss_fn = CrossEntropyLoss()
