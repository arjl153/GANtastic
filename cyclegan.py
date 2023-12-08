import numpy as np
# import pandas as pd
from random import random
import os
import librosa
import soundfile as sf
# import IPython
# from scipy.linalg import svd
from numpy import zeros, ones
from numpy.random import randint
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Layer, InputSpec, Input
from keras import initializers, regularizers, constraints, backend as K
import warnings
warnings.filterwarnings("ignore")
# from matplotlib import pyplot

# Instance normalizer
# Refrenced from: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py

print(tf.config.list_physical_devices())

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
#   print('Device', physical_devices[0])
# except:
#   print(physical_devices, 'no auto growth')

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# Patch GAN discriminator

"""In a traditional GAN, the discriminator typically outputs a single value indicating whether the entire input image is real or fake. In contrast, a PatchGAN discriminates at the patch level, providing a spatially detailed assessment of the realism of different regions in the input image.
The PatchGAN discriminator produces a grid or map of output values, where each value corresponds to the realism of a local patch in the input image. This approach is especially useful for tasks like image-to-image translation, where the goal is to generate realistic high-resolution output images."""

def create_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = tf.image.resize(d, size=(165, 165), method=tf.image.ResizeMethod.BICUBIC)

    
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002), loss_weights=[0.5])
    """The loss for the discriminator is weighted by 50% for each model update.
    This slows down changes to the discriminator relative to the generator model during training."""
    return model


# ResNet block
def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Concatenate()([g, input_layer])
    return g

def create_generator(image_shape, n_resnet=3):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     for _ in range(n_resnet):
#         g = resnet_block(256, g)
    g = Conv2DTranspose(128, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2DTranspose(64, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(1, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # using tanh as it has bigger gradient, helps againt vanishing gradients
    model = Model(in_image, out_image)
    return model

# Composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    output_f = g_model_2(gen1_out)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(learning_rate=0.0002))
    return model

# make pixel between -1 and 1 as tanh is used in generator
def preprocessing(df):
    a, b  = df[0], df[1]
    xl, yl = [], []
    for x in a:
        # xl.append(np.clip((x - x.mean()) / (4 * x.std()), -1, 1))
        xl.append((x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1)
    for y in b:
        # yl.append(np.clip((y - y.mean()) / (4 * y.std()), -1, 1))
        yl.append((y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1)
    xl, yl = np.array(xl), np.array(yl)

    return [xl, yl]
    # a = (a * 2 / s_rate) * 2 - 1
    # b = (b * 2 / s_rate) * 2 - 1
    # return [a, b]

# Generate real samples
def generate_real_samples(df, n_samples, patch_shape):
    print('enter real')
    ix = randint(0, df.shape[0], n_samples)
    print(ix.shape, df.shape, n_samples)
    X = df[ix]
    print(X.shape)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# Generate fake samples
def generate_fake_samples(g_model, df, patch_shape):
    print('enter fake')
    X = g_model.predict(df)
    print('X predicted', X.shape)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    print('y formed', y.shape)
    return X, y

# Save generator models
def save_models():
    g_model_AtoB.save('./trained/g_model_AtoB_gpu.h5')
    g_model_BtoA.save('./trained/g_model_BtoA_gpu.h5')
    d_model_A.save('./trained/d_model_A_gpu.h5')
    d_model_B.save('./trained/d_model_B_gpu.h5')
    c_model_AtoB.save('./trained/c_model_AtoB_gpu.h5')
    c_model_BtoA.save('./trained/c_model_BtoA_gpu.h5')

genre_path_1 = './data/genres_original/classical'
cls_names = sorted([os.path.join(genre_path_1, f) for f in os.listdir(genre_path_1) if os.path.isfile(os.path.join(genre_path_1, f))])

genre_path_2 = './data/genres_original/jazz'
jzz_names = sorted([os.path.join(genre_path_2, f) for f in os.listdir(genre_path_2) if os.path.isfile(os.path.join(genre_path_2, f)) \
                 and 'jazz.00054.wav' not in f])

# genre_path_3 = './data/genres_original/country'
# ctry_names = sorted([os.path.join(genre_path_3, f) for f in os.listdir(genre_path_3) if os.path.isfile(os.path.join(genre_path_3, f))])

s_rate = 22050
n_fft = 2048
hop_length = 512

def load_and_process_audio(speech_names):
    data = [load_audio(x) for x in speech_names]
    max_t = max([x.shape[1] for x in data])
    data = [pad_zeros(x, max_t) for x in data]
    data = np.array(data)
    # data = np.transpose(data, (0, 2, 1))
    print('data shape', data.shape)
    return data

def load_audio(speech_name):
    s, sr = librosa.load(speech_name, sr=s_rate)
    # sr = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    y = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=128)
    # return np.abs(sr)
    return y

def pad(data, T):
    data = [pad_zeros(x, T) for x in data]
    return np.array(data)

def pad_zeros(sr, T):
    return np.pad(sr, ((0, 0), (0, T - sr.shape[1])), mode='constant')

cls = load_and_process_audio(cls_names)
jzz = load_and_process_audio(jzz_names)
# ctry = load_and_process_audio(ctry_names)

# max_t = max([x[2] for x in [cls.shape, jzz.shape, ctry.shape]])
max_t = max([x[2] for x in [cls.shape, jzz.shape]])
cls = pad(cls, max_t)
# max_t = max([x[2] for x in [cls.shape, jzz.shape, ctry.shape]])
max_t = max([x[2] for x in [cls.shape, jzz.shape]])
jzz = pad(jzz, max_t)
# max_t = max([x[2] for x in [cls.shape, jzz.shape, ctry.shape]])
# ctry = pad(ctry, max_t)
cls = np.transpose(cls, (0, 2, 1))
jzz = np.transpose(jzz, (0, 2, 1))
# ctry = np.transpose(ctry, (0, 2, 1))
cls_s = cls[np.random.choice(cls.shape[0], size=50, replace=False)]
jzz_s = jzz[np.random.choice(jzz.shape[0], size=50, replace=False)]
# jzz_s = jzz[np.random.choice(jzz.shape[0], size=25, replace=False)]
# ctry_s = ctry[np.random.choice(ctry.shape[0], size=25, replace=False)]

# datay = np.concatenate(( jzz_s, ctry_s), axis=0)
# datay = np.expand_dims(jzz_s, axis=0)
# datay = jzz_s
# train_df = [datay, cls_s]
print('jazz, cls shapes', jzz_s.shape, cls_s.shape)
train_df = [jzz_s, cls_s]

# load data as train_df
train_df = preprocessing(train_df)
train_df = np.array(train_df)
print('train_df', train_df.shape)
train_df = np.expand_dims(train_df,axis = -1)

def update_image_pool(pool, images, max_size=50):
    selected = []
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)

def generate_fake_samples_BtoA(df, patch_shape):
    print('enter fake B to A')
    X = g_model_BtoA.predict(df)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    print('exit fake B to A')
    return X, y

def generate_fake_samples_AtoB(df, patch_shape):
    print('enter fake A to B')
    X = g_model_AtoB.predict(df)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    print('exit fake A to B')
    return X, y

def train(epochs):
    n_epochs, n_batch = epochs, 1
    n_patch = d_model_A.output_shape[1]
    trainA, trainB = train_df
    poolA, poolB = [], []
    bat_per_epo = len(trainA) // n_batch
    n_steps = bat_per_epo * n_epochs
    
    for i in range(n_steps):
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        # print('real A')
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # print('real B')
        # X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        # print('fake A')
        # X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # print('fake B')
        X_fakeA, y_fakeA = generate_fake_samples_BtoA(X_realB, n_patch)
        # print('fake A')
        X_fakeB, y_fakeB = generate_fake_samples_AtoB(X_realA, n_patch)
        # print('fake B')

        print(X_realA.shape, y_realA.shape, X_realB.shape, y_realB.shape, X_fakeA.shape, y_fakeA.shape, X_fakeB.shape, y_fakeB.shape)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        
        print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        
    # if (i+1) % (bat_per_epo * 10) == 0:
    save_models()

image_shape = train_df[0].shape[1:]
# image_shape = (None, train_df.shape[2], train_df.shape[3])
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)
# with strategy.scope():
g_model_AtoB = create_generator(image_shape)
g_model_BtoA = create_generator(image_shape)
d_model_A = create_discriminator(image_shape)
d_model_B = create_discriminator(image_shape)
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

#call for train fuction
train(epochs=5)


