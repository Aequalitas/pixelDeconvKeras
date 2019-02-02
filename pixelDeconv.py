# Keras implementation of the pixel deconvolution described in 
# [Pixel Deconvolutional Networks] (https://arxiv.org/abs/1705.06820)
# as in a conversion of this Tensorflow implementation https://github.com/divelab/PixelTCN/blob/master/utils/pixel_dcn.py

from keras import backend as K
from keras.layers import Layer, Add
import numpy as np

class pixelDeconv(Layer):

    def __init__(self, output_dim, **kwargs, kernel_size=3):
        self.output_dim = output_dim
		self.kernel_size = kernel_size
        super(pixelDeconv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(pixelDeconv, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		axis = (1,2) # height, widtt -> y,x
		conv0 = K.conv2d(x, self.kernel, padding="same")
		conv1 = K.conv2d(conv0, self.kernel, padding="same")
		dilatedConv0 = dilateTensor(conv0.output, axis, 0, 0)
		dilatedConv1 = dilateTensor(conv1.output, axis, 1, 1)
		conv1 = Add([dilatedConv0, dilatedConv1])
		
		
		shape = list(self.kernel_size) + [self.out_dim, self.out_dim]
		weights = K.truncated_normal(shape)
		conv2 = K.conv2d(conv1, weights, padding="same")
		
		output = Add([conv1, conv2])
		output = K.relu(output)
		
		return output
		
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
		

def kerasUnstack(tensor, axis):
	r = K.shape(tensor)[axis]
	unstackedTensorList = []
	for i in range(r):
		unstackedTensorList.append(tensor)
	return unstackedTensorList

def dilateTensor(input, axis, row_shift, column_shift):
	rows = kerasUnstack(inputs, axis[0])
    row_zeros = K.zeros(rows[0], dtype=np.float32)
    for index in range(len(rows), 0, -1):
        rows.insert(index-row_shift, row_zeros)
    inputs = K.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    columns = kerasUnstack(inputs, axis[1])
    columns_zeros = K.zeros(columns[0], dtype=np.float32)
    for index in range(len(columns), 0, -1):
        columns.insert(index-column_shift, columns_zeros)
    inputs = K.stack(columns, axis=axis[1])
	return inputs
