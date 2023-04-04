import numpy as np
import scipy.signal as signal
import pickle
import cv2

# activation function

class Relu():
    """
    Relu activation layer
    x: input of shape(Samples, Channel, High, Width)
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
    x: input of shape(Samples, Channel, High, Width)
    """
    def __init__(self):
        self.cache = None
        
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.cache = X
        
        return out
    
    def backward(self, backout):
        X = self.cache
        dX = backout * X * (1 - X) # Sigmoid backward
        
        return dX

class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        self.cache = None

    def forward(self, X):
        maxes = np.amax(X, axis = 1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis = 1).reshape(Y.shape[0], 1)
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
        prod = softmax.forward(Y_pred)
        loss = NLLLoss(prod, Y_true)
        Y_serial = np.argmax(Y_true, axis = 1)
        backout = prod.copy()
        backout[np.arange(N), Y_serial] -= 1

        return loss, backout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        loss = NLLLoss(prod, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        backout = prod.copy()
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
    def backward(self, dout):
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
    def __init__(self, N, D_in, H, D_out, weights = ''):
        self.FC1 = FC(D_in, H)
        self.Relu1 = Relu()
        self.FC2 = FC(H, D_out)

        if weights == '':
            pass
        else:
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1.forward(X)
        a1 = self.Relu1.forward(h1)
        h2 = self.FC2.forward(a1)

        return h2

    def backward(self, backout):
        backout = self.FC2.backward(backout)
        backout = self.Relu1.forward(backout)
        backout = self.FC1.backward(backout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

class LetNet5(Net):
    """
    LeNet5
    """
    def __init__(self):
        self.conv1 = Conv(1, 6, 5)






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
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        
        return out
    
    def backward(self, backout): # 這部分不太了解
        X = self.cache
        dX = np.dot(backout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.reshape[0], np.prod(X.shape[1:])).T, backout)
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
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/Cin), (filter_numbers, Cin, filter_size, filter_size)), # Xavier Initialization
                  'grad': 0} # Convolutional layer的 Weight就是filter，可利用相同的filter使weight的parameters大幅降低
        self.b = {'val': np.random.randn(filter_numbers),
                  'grad': 0}
        self.cache = None
        self.pad = padding

    def _forward(self, X):
        X = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')
        (S, Cin, H, W) = X.shape # (Samples, Chennal, High, Width)
        H_ = (H - self.F)/self.S + 1 # 經過Convolutional運算後的High
        W_ = (W - self.F)/self.S + 1 # 經過Convolutional運算後的Width
        Y = np.zeros((S, self.filter_numbers, H_, W_))

        for s in range(S):
            for c in range(self.filter_numbers):
                for h in range(H_):
                    for w in range(W_):
                        Y[s, c, h, w] = np.sum(X[s, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]

        self.cache = X
        return Y

    def _backward(self, backout): # 了解到這
        # backout (S,filter_numbers,H_,W_)
        # W (filter_numbers, Cin, F, F)
        X = self.cache
        (S, Cin, H, W) = X.shape
        H_ = (H - self.F)/self.S + 1
        W_ = (W - self.F)/self.S + 1
        W_rot = np.rot90(np.rot90(self.W['val'])) # 將array逆時針旋轉90度

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['val'].shape)
        db = np.zeros(self.b['val'].shape)

        # dW
        for fn in range(self.filter_numbers):
            for ci in range(Cin):
                for h in range(self.F):
                    for w in range(self.F):
                        dW[fn, ci, h, w] = np.sum(X[:, ci, h:h+H_, w:w+W_] * backout[:, fn, :, :])

        # db
        for fn in range(self.filter_numbers):
            db[fn] = np.sum(backout[:, fn, :, :])

        dout_pad = np.pad(backout, ((0, 0), (0, 0), (self.F, self.F), (self.F, self.F)), 'constant')
        #print("dout_pad.shape: " + str(dout_pad.shape))
        # dX
        for s in range(S):
            for ci in range(Cin):
                for h in range(H):
                    for w in range(W):
                        #print("self.F.shape: %s", self.F)
                        #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                        dX[s, ci, h, w] = np.sum(W_rot[:, ci, :, :] * dout_pad[n, :, h:h+self.F, w:w+self.F])

        return dX

# 缺Maxpool、LeNet、SGDMomentum、
# data的tensor順序(S, H, W, Cin)與模型不符(S, Cin, H, W)，需更改
# LeNet5與TwoLayerNet.py是否分開