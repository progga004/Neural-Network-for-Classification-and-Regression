from tinyNeuralNet import *
import scipy.io as sio
import scipy
import pickle

X = np.array([[-1,2,-3.],[4.,-5/2,6/2],[7/2,8/2,-9/2],[.1,.3,.1],[-.1,8,-10],[-.1,8,-10]])
X2 = np.array([[4.,-5/2,6/2],[7/2,8/2,-9/2],[.1,.3,.1],[0,1,0],[9,99,1]])

X2 = np.vstack([X,X2])

y = np.array([1,-1,1,1,-1,1])
z = np.array([-.05,-8,0,1.23,4.56,7.89])
theta = np.array([10,-.4,-.02])
y2 = np.array([1,1,1,1,-1,-1,-1,-1,1,-1,-1])
              
import sys

s = sys.argv[1]
if s == 'check_sigmoid':
    zz = sigmoid_fn(z)
    zz = np.round(zz*10000)/10000
    assert(np.sum(zz)==3.7508)
    
    
elif s == 'check_relu':
    relu = ReLU()
    print('checking params')
    
    print('checking forward')
    zz = relu.forward(z)
    assert(np.sum(zz)==13.68)
    
    print('checking backward')
    zz = relu.backward(X,X*2)
    zz = np.round(np.sum(zz)*10000)/10000
    assert(zz==33.0)
    
    

    
elif s == 'check_linear':
    lin = Linear(11,6,1)
    print('checking params')
    lin.W = np.ones(lin.W.shape)
    lin.W[0,:] = range(6)
    lin.W[:,4] = range(11)
    lin.b = -np.ones(lin.b.shape)
    lin.b[0] = -3.14
    
    print('checking forward')
    zz = lin.forward(np.vstack([z,z]).T)
    zz = np.sum(zz)
    zz = np.round(zz*10000)/10000
    assert(zz==567.0)
    
    
    print('checking backward')
    zz = lin.backward(X2/2,X2)
    zz = np.sum(zz)
    zz = np.round(zz*10000)/10000
    assert(zz==865.55)
          
        
    print('checking update_var')
    lin.update_var(X2/2,X,2.34)
    zz = np.sum(lin.W)
    zz = np.round(zz*100)/100
    assert(zz==-3581.2)
    
    zz = np.sum(lin.b)
    zz = np.round(zz*100)/100
    assert(zz==-153.31)
    
    
    
        
elif s == 'CrossEntropyLoss':
    
    loss = CrossEntropyLoss()
    
    y = np.array([[0,1,0],[1,0,0],[1,0,0],[0,0,1]])
    yhat = np.array([[.5,.1,.2],[.3,.4,.5],[.9,.1,.3],[0,0,1]])
                  
                 
    zz = loss.forward(y,yhat)
    zz = np.sum(zz)
    zz = np.round(zz*10000)/10000
    assert(zz == 3.72570)
    zz = loss.backward(y,yhat)
    zz = np.sum(zz)
    zz = np.round(zz*10000)/10000
    assert(zz == 0.)
    
    
    
    
elif s == 'tiny_model':
    y = np.array([[0,1,0],[1,0,0],[1,0,0],[0,0,1]])
    
    
    class tinyModel(Model):
        def __init__(self):
            self.layers = [Linear(7,6,1),ReLU(), Linear(4,7,1)]
            self.loss_fn = CrossEntropyLoss()

    model = tinyModel()
    
    for layer in model.layers:
        if 'W' in dir(layer):
            layer.W = np.reshape(range(np.prod(layer.W.shape)),layer.W.shape)/30+.9
            layer.b = np.array(range(len(layer.b)))*(-1.)-2
            
            
    a,b = model.forward(X,y)
    a = np.round(a * 10000)/10000
    assert(a==755.524 )
    
    
    b = np.sum(np.log(np.abs(b)+1))
    b = np.round(b * 10000)/10000
    assert(b==43.148)
    
       
    
elif s == 'tiny_model_backward':
    y = np.array([[0,1,0],[1,0,0],[1,0,0],[0,0,1]])
    
    
    class tinyModel(Model):
        def __init__(self):
            self.layers = [Linear(7,6,1),ReLU(), Linear(4,7,1)]
            self.loss_fn = CrossEntropyLoss()

    model = tinyModel()
    
    for layer in model.layers:
        if 'W' in dir(layer):
            layer.W = np.reshape(range(np.prod(layer.W.shape)),layer.W.shape)/30+.9
            layer.b = np.array(range(len(layer.b)))*(-1.)-2
            
            
    model.forward(X,y)
    model.backward(X,y)
    
    sg = [np.linalg.norm(aa) for aa in model.states_grad]
    sg = [np.round(aa*10000)/10000 for aa  in sg]
    assert(sg==[13.6731, 14.395, 2.4495] )
    
    
    
else:
    print('no such command:',s)
    
print('passed!')
