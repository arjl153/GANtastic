import numpy as np
import pandas as pd
from matplotlib import pyplot
from random import random
import os
from scipy.linalg import svd
from numpy import load, zeros, ones, asarray, vstack
from numpy.random import randint
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, load_model
import keras.layers as layers
from keras.layers import Conv2D, Conv2DTranspose,BatchNormalization, LeakyReLU, Activation, Concatenate, Layer, InputSpec, Input, Conv1D, Conv1DTranspose
from keras import initializers, regularizers, constraints, backend as K
import tqdm


### Model definition

def create_discriminator(input_shape):
    init = RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=0.0002), loss_weights=[0.5])
    return model


def create_generator(input_shape):
    init = RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (7, 7), padding='same', kernel_initializer=init, input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(1, (7, 7), padding='same', kernel_initializer=init))
    model.add(layers.Activation('tanh'))
    return model

### Loading pre-processed data

genre_path_1, genre_path_2  = r'JC_C/train' , r'JC_J/train'
classic_names = sorted([os.path.join(genre_path_1, f) for f in os.listdir(genre_path_1) if os.path.isfile(os.path.join(genre_path_1, f))])
jazz_names = sorted([os.path.join(genre_path_2, f) for f in os.listdir(genre_path_2) if os.path.isfile(os.path.join(genre_path_2, f))])

classic_array = np.array([np.load(x) for x in classic_names[:10000]])
jazz_array = np.array([np.load(x) for x in jazz_names[:10000]])

classic_array = classic_array.reshape((-1,1,64,84,1))
jazz_array = jazz_array.reshape((-1,1,64,84,1))

dataset = np.array(list(zip(classic_array,jazz_array)))
dataset = tf.convert_to_tensor(dataset)


### Model Definition and Training
    
input_shape = dataset[0].shape[2:]
epoch_decay, len_dataset = 100, 10000 
epochs = 300

bce = tf.losses.BinaryCrossentropy(from_logits=True)
cycle_loss_function = tf.losses.MeanAbsoluteError()
identity_loss_function = tf.losses.MeanAbsoluteError()

def generator_adversarial_function(fakes):
    return bce(1, fakes)


def discriminator_loss_function(reals, fakes):
    return bce(1, reals), bce(0, fakes)


# Using ItemPool for stable training. Refrenced from https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/data.py. 
class ItemPool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []
    def __call__(self, in_items):
        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

A2B_pool, B2A_pool = ItemPool(50), ItemPool(50)

generator_a_to_b, discriminator_b = create_generator(input_shape), create_discriminator(input_shape)
generator_b_to_a, discriminator_a = create_generator(input_shape), create_discriminator(input_shape)


generator_learning_rate_decay = LinearDecay(0.0002, epochs * len_dataset, epoch_decay * len_dataset)
discriminator_learning_rate_decay = LinearDecay(0.0002, epochs * len_dataset, epoch_decay * len_dataset)

generator_optimizer = keras.optimizers.Adam(learning_rate=generator_learning_rate_decay, beta_1=.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=discriminator_learning_rate_decay, beta_1=0.5)


identity_loss_weight, cycle_loss_weight = 5, 10

@tf.function
def train_generator(A, B):
    with tf.GradientTape() as t:
        a_to_b_generated = generator_a_to_b(A, training=True)
        b_to_a_generated = generator_b_to_a(B, training=True)

        a_cycle_generated = generator_b_to_a(a_to_b_generated, training=True)
        b_cycle_generated = generator_a_to_b(b_to_a_generated, training=True)

        a_identity = generator_b_to_a(A, training=True)
        b_identity = generator_a_to_b(B, training=True)

        discriminator_b_preds_on_generated = discriminator_b(a_to_b_generated, training=True)
        discriminator_a_preds_on_generated = discriminator_a(b_to_a_generated, training=True)

        a_to_b_generator_loss = generator_adversarial_function(discriminator_b_preds_on_generated)
        b_to_a_generator_loss = generator_adversarial_function(discriminator_a_preds_on_generated)

        a_cycle_loss = cycle_loss_function(A, a_cycle_generated)
        b_cycle_loss = cycle_loss_function(B, b_cycle_generated)

        a_identity_loss = identity_loss_function(A, a_identity)
        b_identity_loss = identity_loss_function(B, b_identity)

        generator_loss = (a_to_b_generator_loss + b_to_a_generator_loss) + (a_cycle_loss + b_cycle_loss) * cycle_loss_weight + (a_identity_loss + b_identity_loss) * identity_loss_weight

    generator_gradient = t.gradient(generator_loss, generator_a_to_b.trainable_variables + generator_b_to_a.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradient, generator_a_to_b.trainable_variables + generator_b_to_a.trainable_variables))

    return a_to_b_generated, b_to_a_generated, {'a_to_b_generator_loss': a_to_b_generator_loss,
                      'b_to_a_generator_loss': b_to_a_generator_loss,
                      'a_cycle_loss': a_cycle_loss,
                      'b_cycle_loss': b_cycle_loss,
                      'a_identity_loss': a_identity_loss,
                      'b_identity_loss': b_identity_loss}


@tf.function
def train_discriminator(A, B, a_to_b_generated, b_to_a_generated):
    with tf.GradientTape() as t:
        discriminator_a_preds_on_a = discriminator_a(A, training=True)
        discriminator_a_preds_on_generated = discriminator_a(b_to_a_generated, training=True)
        discriminator_b_preds_on_b = discriminator_b(B, training=True)
        discriminator_b_preds_on_generated = discriminator_b(a_to_b_generated, training=True)

        discriminator_a_real_loss, discriminator_a_fake_loss = discriminator_loss_function(discriminator_a_preds_on_a, discriminator_a_preds_on_generated)
        discriminator_b_real_loss, discriminator_b_fake_loss = discriminator_loss_function(discriminator_b_preds_on_b, discriminator_b_preds_on_generated)
        discriminator_loss = (discriminator_a_real_loss + discriminator_a_fake_loss) + (discriminator_b_real_loss + discriminator_b_fake_loss) 
    discriminator_gradient = t.gradient(discriminator_loss, discriminator_a.trainable_variables + discriminator_b.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator_a.trainable_variables + discriminator_b.trainable_variables))

    return {'discriminator_a_real_loss': discriminator_a_real_loss + discriminator_a_fake_loss,
            'discriminator_b_real_loss': discriminator_b_real_loss + discriminator_b_fake_loss}



def train_step(A, B):
    a_to_b_generated, b_to_a_generated, generator_losses = train_generator(A, B)
    a_to_b_generated, b_to_a_generated = A2B_pool(a_to_b_generated), B2A_pool(b_to_a_generated)  
    discriminator_losses = train_discriminator(A, B, a_to_b_generated, b_to_a_generated)
    return generator_losses, discriminator_losses

for ep in tqdm.trange(epochs):
        
    for A, B in tqdm.tqdm(dataset, total=len_dataset):
        generator_losses, discriminator_losses = train_step(A, B)

    if ep%10 == 0:
        generator_a_to_b.save('new_trained/gatob_{}_model.h5'.format(ep+1))
        generator_b_to_a.save('new_trained/gbtoa_{}_model.h5'.format(ep+1))
        discriminator_a.save('new_trained/db_{}_model.h5'.format(ep+1))
        discriminator_b.save('new_trained/da_{}_model.h5'.format(ep+1))

# generator_a_to_b = load_model('trained/gatob_300_model.h5')


