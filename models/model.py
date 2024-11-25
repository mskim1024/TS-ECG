import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(1234)   # TensorFlow 난수 시드 설정
    
class ResidualConnection(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, name, **kargs):
        super(ResidualConnection, self).__init__()
        
        self.conv = layers.Conv1D(filters=kargs[filters], kernel_size=kargs[kernel_size], 
                                  dilation_rate=dilation_rate, padding='causal', name=name)
        self.batch_norm = layers.BatchNormalization()
        self.elu = layers.ELU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.elu(x)
        
        return layers.Add()([x, inputs])

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, idx, num_modules=4, **kargs):
        super(ResidualBlock, self).__init__()
        
        filters = [f'conv{i+1}_filter' for i in range(num_modules-1)]
        filters.append('conv1x1_filter')
        kernel_sizes = [f'conv{i+1}_kernel_size' for i in range(num_modules-1)]
        kernel_sizes.append('conv1x1_kernel_size')

        self.idx = idx
        self.residual_connections = [ResidualConnection(filters=filters[i], kernel_size=self.kernel_size[i], idx=idx,
                                                        dilation_rate=kargs['dilation_rate']**(i+1), name=f'conv{idx}_{i+1}')
                                    for i in range(4)]

    def call(self, inputs):
        x = inputs
        for residual_connection in self.residual_connections:
            x = residual_connection(x)
        
        return layers.Add()([x, inputs])

class BiLSTMModule(tf.keras.layers.Layer):
    def __init__(self, unit, name, **kargs):
        super(BiLSTMModule, self).__init__()
        
        self.bilstm = layers.Bidirectional(layers.LSTM(kargs[unit], return_sequences=True), name=name)
        self.batch_norm = layers.BatchNormalization()
        self.elu = layers.ELU()

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.batch_norm(x)
        x = self.elu(x)
        return x

class BiLSTMBlock(tf.keras.layers.Layer):
    def __init__(self, idx, num_modules=4, **kargs):
        super(BiLSTMBlock, self).__init__()

        units = [f'lstm{i+1}_units' for i in range(num_modules)]
        layer_name = f'bi-lstm_{idx}'
        self.bilstm_modules = [BiLSTMModule(units[i], layer_name, kargs) for i in range(num_modules)]

    def call(self, inputs):
        x = inputs
        for module in self.bilstm_modules:
            x = module(x)
        return layers.Add()([x, inputs])  # Residual connection


class BuildModel(tf.keras.Model):
    def __init__(self, **kargs):
        super(BuildModel, self).__init__()
        self.kargs = kargs
        self.residual_blocks = [ResidualBlock(i+1, num_modules=4, **kargs) for i in range(4)]
        self.bilstm_net = [BiLSTMBlock(i+1, num_modules=4, **kargs) for i in range(4)]

        self.global_avg_pooling = layers.GlobalAveragePooling1D()

        self.dense_relu1 = layers.Dense(kargs['fc1'], activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense_relu2 = layers.Dense(kargs['fc2'], activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.outputs_layer = layers.Dense(kargs['num_classes'], activation='softmax')

    def call(self, inputs):
        cnn_inputs, lstm_inputs = inputs
        
        x = cnn_inputs
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        cnn_outputs = self.global_avg_pooling(x)

        y = lstm_inputs
        y = layers.MultiHeadAttention(key_dim=512, 
                                      num_heads=8, 
                                      dropout=0.25)(y, y)
        for bilstm in self.bilstm_net:
            y = bilstm(y)
        
        lstm_outputs = y[:, -1, :]
        z = layers.concatenate([cnn_outputs, lstm_outputs])

        z = self.dense_relu1(z)
        z = self.dropout1(z)
        z = self.dense_relu2(z)
        z = self.dropout2(z)
        outputs = self.outputs_layer(z)
        
        return outputs