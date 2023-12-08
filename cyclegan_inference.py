import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Layer, InputSpec, Input
from keras import initializers, regularizers, constraints, backend as K
import numpy as np
import librosa
import soundfile as sf
import os
import warnings
warnings.filterwarnings("ignore")

s_rate = 22050
n_fft = 2048
hop_length = 512

class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

g_AtoB = tf.keras.models.load_model('./trained/g_model_AtoB_gpu.h5', custom_objects={"InstanceNormalization": InstanceNormalization})
g_BtoA = tf.keras.models.load_model('./trained/g_model_BtoA_gpu.h5', custom_objects={"InstanceNormalization": InstanceNormalization})

def load_audio(speech_name):
    s, sr = librosa.load(speech_name, sr=s_rate)
    # sr = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    y = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=128)
    print('mel', y)
    return y

def convert_to_audio(signal):
    signal = signal[0, :, :, 0]
    signal = ((signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * (jz1_max - jz1_min) + jz1_min)
    print('sig', signal.shape)
    cls1_abs = np.transpose(signal)
    # signal = (signal + 1) / 2.0
    # cls1 = np.multiply(np.divide(jz1, jz1_abs), cls1_abs)
    # cls1 = librosa.istft(cls1, n_fft=n_fft, hop_length=hop_length)
    # return cls1
    y = librosa.feature.inverse.mel_to_audio(cls1_abs, sr=s_rate, hop_length=hop_length, n_fft=n_fft)
    return y


# genre_path_1 = './data/genres_original/classical'
# cls_names = sorted([os.path.join(genre_path_1, f) for f in os.listdir(genre_path_1) if os.path.isfile(os.path.join(genre_path_1, f))])

jz1 = load_audio('./data/genres_original/jazz/jazz.00000.wav')
jz1 = np.pad(jz1, ((0, 0), (0, 1314 - jz1.shape[1])), mode='constant')
# jz1_abs = np.abs(jz1)
jz1_at = np.transpose(jz1)


jz1_min = np.min(jz1)
jz1_max = np.max(jz1)

jz1_scaled = (jz1_at - jz1_min) / (jz1_max - jz1_min) * 2 - 1


jz1_at = np.expand_dims(jz1_scaled, axis=(0,3))
print('jz1', jz1_at.shape)

cls1_at = g_AtoB.predict(jz1_at)
# cls1_at = cls1_at[0, :, :, 0]
# print(cls1_at.shape)
# cls1_abs = np.transpose(cls1_at)
# cls1 = np.multiply(np.divide(jz1, jz1_abs), cls1_abs)

# cls1 = librosa.istft(cls1, n_fft=n_fft, hop_length=hop_length)
print('cls1', cls1_at.shape)
cls1 = convert_to_audio(cls1_at)
sf.write('2gan-convert.wav', cls1, s_rate)

back = g_BtoA.predict(cls1_at)
back = convert_to_audio(back)
sf.write('2gan-cycle.wav', cls1, s_rate)


# cls2 = load_audio('./data/genres_original/classical/classical.00000.wav')
# cls2 = np.pad(cls2, ((0, 0), (0, 1314 - cls2.shape[1])), mode='constant')
# cls2_abs = np.abs(cls2)
# cls2_at = np.transpose(cls2_abs)

# cls2_at = np.expand_dims(cls2_at, axis=(0,3))
# print(cls2_at.shape)

# unknown = g_AtoB.predict(cls2_at)
# unknown = convert_to_audio(unknown)
# sf.write('2unknown.wav', unknown, s_rate)

