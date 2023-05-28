import numpy as np
import scipy.signal as signal
import pickle
import random
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

# activation function

class ReLu():
    """
    Relu activation layer
    x: input of shape(Samples, High, Width, Channel)
    """
    def __init__(self):
        self.cache = None
        
    def forward(self, X):
        out = np.maximum(0, X)
        self.cache = X
        
        return out
    
    def backward(self, backout): # 這部分不太了解
        X = self.cache
        dX = np.array(backout, copy = True)
        dX[X <= 0] = 0
        
        return dX
        
class Sigmoid():
    """
    Sigmoid activation layer
    x: input of shape(Samples, High, Width, Channel)
    """
    def __init__(self):
        self.cache = None
        
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.cache = X
        
        return out
    
    def backward(self, backout):
        X = self.cache
        dX = backout * 1 / (1 + np.exp(-X)) * (1 - 1 / (1 + np.exp(-X))) # Sigmoid backward
        
        return dX

class Swish():
    """
    Swish activation layer
    x: input of shape(Samples, High, Width, Channel)
    """

    def __init__(self):
        self.cache = None

    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        out = X * out
        self.cache = X

        return out

    def backward(self, backout):
        X = self.cache
        dX = backout * 1 / (1 + np.exp(-X))  + X * 1 / (1 + np.exp(-X)) * (1 - 1 / (1 + np.exp(-X)))  # Sigmoid backward

        return dX


class Softmax():
    """
    Softmax_Improved activation layer
    """

    def __init__(self):
        self.cache = None

    def forward(self, X):
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)

        return Z

    def backward(self, backout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(Z):
            i = np.argmax(Z[n])
            dZ[n, :] = np.diag(Z[n]) - np.outer(Z[n], Z[n])
            M = np.zeros((N, N))
            M[:, i] = 1
            dY[n, :] = np.eye(N) - M
        dX = np.dot(backout, dZ)
        dX = np.dot(dX, dY)

        return dX

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred * Y_true, axis = 1)
    for e in M:
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)

    return loss/N


class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax.forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        backout = prob.copy()
        backout[np.arange(N), Y_serial] -= 1
        return loss, backout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        loss = NLLLoss(Y_pred, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        backout = Y_pred.copy()
        backout[np.arange(N), Y_serial] -= 1

        return loss, backout

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, backout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class TwoLayerNet(Net):
    """
    Simple 2 layer NN
    """
    def __init__(self, S, D_in, H, D_out, weights = ''):
        self.FC1 = FC(D_in, H)
        self.ReLu1 = ReLu()
        self.FC2 = FC(H, D_out)

        if weights == '':
            pass
        else:
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1.forward(X)
        a1 = self.ReLu1.forward(h1)
        h2 = self.FC2.forward(a1)

        return h2

    def backward(self, backout):
        backout = self.FC2.backward(backout)
        backout = self.ReLu1.forward(backout)
        backout = self.FC1.backward(backout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

 # layer

class FC():
    """
    Fully Connected layer
    """
    def __init__(self, D_in, D_out):
        self.cache = None
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in, D_out)),  # 使用He Weight initialization
                  'grad': 0} # gradient初值為0
        self.b = {'val': np.random.randn(D_out),
                  'grad': 0}
        
    def forward(self, X):
        S = X.shape[0]
        X_new = X.reshape(S, -1)
        # print('X_new.shape',X_new.shape)
        # print("self.W['val'].shape",self.W['val'].shape)
        out = np.dot(X_new, self.W['val']) + self.b['val']
        self.cache = X_new
        
        return out
    
    def backward(self, backout): # 這部分不太了解
        X_new = self.cache
        dX = np.dot(backout, self.W['val'].T).reshape(X_new.shape)
        self.W['grad'] = np.dot(X_new.reshape(X_new.shape[0], np.prod(X_new.shape[1:])).T, backout)
        self.b['grad'] = np.sum(backout, axis = 0)
        
        return dX
    
    def updata_params(self, lr=.001):
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']
        
        
        
        

class Conv():
    """
    Convolutional layer
    """
    def __init__(self, filter_size, Cin, filter_numbers, stride=1, padding=0, bias=True):
        self.F = filter_size # filter 的大小
        self.Cin = Cin # 進入Convolutional layer的Channel數
        self.filter_numbers = filter_numbers # filter的數量，即為下一層Conv layer的的Cin
        self.S = stride # filter一次移動的格數
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/Cin), (filter_numbers, filter_size, filter_size, Cin)), # Xavier Initialization
                  'grad': 0} # Convolutional layer的 Weight就是filter，可利用相同的filter使weight的parameters大幅降低
        self.b = {'val': np.random.randn(filter_numbers),
                  'grad': 0}
        self.cache = None
        self.pad = padding

    def forward(self, X):
        X = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')
        (S, H, W, Cin) = X.shape # (Samples, Chennal, High, Width)
        H_ = round((H - self.F)/self.S) + 1 # 經過Convolutional運算後的High
        W_ = round((W - self.F)/self.S) + 1 # 經過Convolutional運算後的Width
        Y = np.zeros((S, H_, W_, self.filter_numbers))

        for s in range(S):
            for c in range(self.filter_numbers):
                for h in range(H_):
                    for w in range(W_):
#                         Y[s, h, w, c] = np.sum(X[s, h:h+self.F, w:w+self.F, :] * self.W['val'][:, :, :, c]) + self.b['val'][c]
                        Y[s, h, w, c] = np.sum((X[s, h:h+self.F, w:w+self.F, :] , self.W['val'][c, :, :, :])) + self.b['val'][c]
        self.cache = X
        return Y

    def backward(self, backout): # 了解到這
        # backout (S,filter_numbers,H_,W_)
        # W (filter_numbers, Cin, F, F)
        X = self.cache
        (S, H, W, Cin) = X.shape
        H_ = round((H - self.F)/self.S + 1)
        W_ = round((W - self.F)/self.S + 1)
        W_rot = np.rot90(np.rot90(self.W['val'])) # 將array逆時針旋轉90度

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['val'].shape)
        db = np.zeros(self.b['val'].shape)

        # dW
        for fn in range(self.filter_numbers):
            for ci in range(Cin):
                for h in range(self.F):
                    for w in range(self.F):
                        dW[fn, h, w, ci] = np.sum(X[:, h:h+H_, w:w+W_, ci] * backout[:, :, :, fn])

        # db
        for fn in range(self.filter_numbers):
            db[fn] = np.sum(backout[:, :, :, fn])

        dout_pad = np.pad(backout, ((0, 0), (0, 0), (self.F, self.F), (self.F, self.F)), 'constant')
        #print("dout_pad.shape: " + str(dout_pad.shape))
        # dX
        for s in range(S):
            for ci in range(Cin):
                for h in range(H):
                    for w in range(W):
                        #print("self.F.shape: %s", self.F)
                        #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                        dX[s, h, w, ci] = np.sum(np.dot(W_rot[:, :, :, ci] , dout_pad[s, h:h+self.F, w:w+self.F, :]))

        return dX

class MaxPool():
    def __init__(self, filter_size, stride):
        self.F = filter_size
        self.S = stride
        self.cache = None

    def forward(self, X):
        # X: (S, Cin, H, W): maxpool along 3rd, 4th dim
        (S, H, W, Cin) = X.shape
        F = self.F
        W_ = int(float(W)/F)
        H_ = int(float(H)/F)
        Y = np.zeros((S, W_, H_, Cin))
        M = np.zeros(X.shape) # mask
        for s in range(S):
            for cin in range(Cin):
                for w_ in range(W_):
                    for h_ in range(H_):
                        Y[s, w_, h_, cin] = np.max(X[s,F*w_:F*(w_+1), F*h_:F*(h_+1), cin])
                        i, j = np.unravel_index(X[s, F*w_:F*(w_+1), F*h_:F*(h_+1), cin].argmax(), (F, F))
                        M[s, F*w_+i, F*h_+j, cin] = 1
        self.cache = M
        return Y

    def backward(self, dout):
        M = self.cache
        (N,H,W,Cin) = M.shape
        dout = np.array(dout)
        #print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
        dX = np.zeros(M.shape)
        for n in range(N):
            for c in range(Cin):
                #print("(n,c): (%s,%s)" % (n,c))
                dX[n,:,:,c] = dout[n,:,:,c].repeat(2, axis=0).repeat(2, axis=1)
        return dX*M


class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho * self.velocities[i] + (1 - self.rho) * self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr * self.velocities[i] + self.reg * self.parameters[i]['val'])


class LeNet5(Net):
    """
    LeNet5
    """
    def __init__(self):
        self.conv1 = Conv(filter_size=5, Cin=3, filter_numbers=6) # 256 - 4 = 252
        self.ReLU1 = ReLu()
        self.Maxpool1 = MaxPool(2,2) # 252 / 2 = 126
        self.conv2 = Conv(filter_size=5, Cin=6, filter_numbers=16) # 126 - 4 = 122
        self.ReLU2 = ReLu()
        self.Maxpool2 = MaxPool(2,2) # 122 / 2 = 61
        self.FC1 = FC(16*61*61, 1200)
        self.ReLU3 = ReLu()
        self.FC2 = FC(1200, 840)
        self.ReLU4 = ReLu()
        self.FC3 = FC(840, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, data):
        h1 = self.conv1.forward(data) # hidden layer
        a1 = self.ReLU1.forward(h1) # activation layer
        p1 = self.Maxpool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.ReLU2.forward(h2)
        p2 = self.Maxpool2.forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(data.shape[0], -1)  # Flatten
        h3 = self.FC1.forward(fl)
        a3 = self.ReLU3.forward(h3)
        h4 = self.FC2.forward(a3)
        a5 = self.ReLU4.forward(h4)
        h5 = self.FC3.forward(a5)
        a5 = self.Softmax.forward(h5)
        return a5

    def backward(self, backout):
        # backout = self.Softmax.backward(backout)
        backout = self.FC3.backward(backout)
        backout = self.ReLU4.backward(backout)
        backout = self.FC2.backward(backout)
        backout = self.ReLU3.backward(backout)
        backout = self.FC1.backward(backout)
        backout = backout.reshape(self.p2_shape)  # reshape
        backout = self.Maxpool2.backward(backout)
        backout = self.ReLU2.backward(backout)
        backout = self.conv2.backward(backout)
        backout = self.Maxpool1.backward(backout)
        backout = self.ReLU1.backward(backout)
        backout = self.conv1.backward(backout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W,
                self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b,
         self.FC3.W, self.FC3.b] = params

class LeNet5_Im(Net):
    """
    LeNet5_Improved
    """
    def __init__(self):
        self.conv1 = Conv(filter_size=3, Cin=3, filter_numbers=6) # 254 - 2 = 252
        self.Swish1 = Swish()
        self.Maxpool1 = MaxPool(2,2) # 252 / 2 = 126
        self.conv2 = Conv(filter_size=3, Cin=6, filter_numbers=16) # 126 - 2 = 124
        self.Swish2 = Swish()
        self.Maxpool2 = MaxPool(2,2) # 124 / 2 = 62
        self.conv3 = Conv(filter_size=3, Cin=16, filter_numbers=26)  # 62 - 2 = 60
        self.Swish3 = Swish()
        self.Maxpool3 = MaxPool(2, 2) # 60 / 2 = 30
        self.FC1 = FC(26*30*30, 1200)
        self.Swish4 = Swish()
        self.FC2 = FC(1200, 840)
        self.Swish5 = Swish()
        self.FC3 = FC(840, 50)
        self.Softmax = Softmax() # 到這

        self.p3_shape = None

    def forward(self, data):
        h1 = self.conv1.forward(data) # hidden layer
        a1 = self.Swish1.forward(h1) # activation layer
        p1 = self.Maxpool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.Swish2.forward(h2)
        p2 = self.Maxpool2.forward(a2)
        h3 = self.conv3.forward(p2)
        a3 = self.Swish3.forward(h3)
        p3 = self.Maxpool3.forward(a3)
        self.p3_shape = p3.shape
        fl = p3.reshape(data.shape[0], -1)  # Flatten
        h3 = self.FC1.forward(fl)
        a4 = self.Swish4.forward(h3)
        h4 = self.FC2.forward(a4)
        a5 = self.Swish5.forward(h4)
        h5 = self.FC3.forward(a5)
        a6 = self.Softmax.forward(h5)
        return a6

    def backward(self, backout):
        # backout = self.Softmax.backward(backout)
        backout = self.FC3.backward(backout)
        backout = self.Swish5.backward(backout)
        backout = self.FC2.backward(backout)
        backout = self.Swish4.backward(backout)
        backout = self.FC1.backward(backout)
        backout = backout.reshape(self.p3_shape)  # reshape
        backout = self.Maxpool3.backward(backout)
        backout = self.Swish3.backward(backout)
        backout = self.conv3.backward(backout)
        backout = self.Maxpool2.backward(backout)
        backout = self.Swish2.backward(backout)
        backout = self.conv2.backward(backout)
        backout = self.Maxpool1.backward(backout)
        backout = self.Swish1.backward(backout)
        backout = self.conv1.backward(backout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W,
                self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b,
         self.FC3.W, self.FC3.b] = params



# 缺Maxpool、LeNet、SGDMomentum (完成)
# data的tensor順序(S, H, W, Cin)與模型不符(S, Cin, H, W)，需更改 (完成)
# LeNet5與TwoLayerNet.py是否分開 (完成)
# data進去的shape與程式要的不同，需修改


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

