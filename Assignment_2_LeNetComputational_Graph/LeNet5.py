import numpy as np
import scipy.signal as signal
import cv2

class Conv2D:
    def __init__(self, num_filters, filter_size, input_channels, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.01
        self.biases = np.zeros((1, num_filters))

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        output_height = int((input_height - self.filter_size + 2 * self.padding) / self.stride) + 1
        output_width = int((input_width - self.filter_size + 2 * self.padding) / self.stride) + 1

        self.outputs = np.zeros((batch_size, output_height, output_width, self.num_filters))

        padded_inputs = np.pad(self.inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        for b in range(batch_size):
            for f in range(self.num_filters):
                # 使用SciPy的convolve2d函数进行卷积
                convolved = signal.convolve2d(padded_inputs[b], np.rot90(self.weights[f], 2), mode='valid')
                self.outputs[b, :, :, f] = convolved[::self.stride, ::self.stride]

            # 添加偏置
            self.outputs[b] += self.biases

        return self.outputs

    def backward(self, d_outputs, learning_rate):
        batch_size, output_height, output_width, num_filters = d_outputs.shape

        padded_inputs = np.pad(self.inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        dpadded_inputs = np.zeros(padded_inputs.shape)
        d_weights = np.zeros(self.weights.shape)
        d_biases = np.zeros(self.biases.shape)

        for b in range(batch_size):
            for f in range(num_filters):
                # 计算权重和偏差的梯度
                d_weights[f, :, :, :] += signal.convolve2d(padded_inputs[b], d_outputs[b, :, :, f], mode='valid')
                d_biases[:, f] += np.sum(d_outputs[b, :, :, f], axis=(0, 1))

                # 计算输入的梯度
                dpadded_inputs[b] += signal.convolve2d(d_outputs[b, :, :, f], self.weights[f], mode='full')

        # 取消填充
        d_inputs = dpadded_inputs[:, self.padding:-self.padding, self.padding:-self.padding, :]

        # 更新参数
        self.weights -= learning_rate * d_weights / batch_size
        self.biases -= learning_rate * d_biases / batch_size

        return d_inputs



class FC:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01  # 初始化权重
        self.biases = np.zeros((1, output_size))  # 初始化偏置

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, d_outputs, learning_rate):
        d_weights = np.dot(self.inputs.T, d_outputs)
        d_biases = np.sum(d_outputs, axis=0, keepdims=True)
        d_inputs = np.dot(d_outputs, self.weights.T)

        # 更新参数
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs