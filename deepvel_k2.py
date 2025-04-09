import numpy as np
import platform
import os
from astropy.io import fits
import time
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

if platform.node() != 'vena':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Add
from tensorflow.keras.models import Model


class deepvel(object):
    def __init__(self, observations, output, border=0):
        """
        Class used to predict horizontal velocities from two consecutive continuum images
        """
        # Only allocate needed memory with TensorFlow
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except:
                pass

        self.border = border
        n_timesteps, nx, ny = observations.shape

        self.n_frames = n_timesteps - 1
        self.nx = nx - 2 * self.border
        self.ny = ny - 2 * self.border

        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1
        self.n_conv_layers = 20
        self.observations = observations
        self.output = output

        print(f"Images without border are of size: {self.nx}x{self.ny}")
        print(f"Number of predictions to be made: {self.n_frames}")

    def residual(self, inputs):
        x = Conv2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, inputs])
        return x

    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv = Conv2D(self.n_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for _ in range(self.n_conv_layers):
            x = self.residual(x)

        x = Conv2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, conv])

        final = Conv2D(6, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)

        self.model = Model(inputs=inputs, outputs=final)
        self.model.load_weights('network/deepvel_weights.hdf5')

    def validation_generator(self):
        self.median_i = np.median(self.observations[:, self.border:-self.border, self.border:-self.border])
        input_validation = np.zeros((self.batch_size, self.nx, self.ny, 2), dtype='float32')

        while True:
            for i in range(self.n_frames):
                input_validation[:, :, :, 0] = self.observations[i*self.batch_size:(i+1)*self.batch_size,
                                                                  self.border:-self.border,
                                                                  self.border:-self.border] / self.median_i
                input_validation[:, :, :, 1] = self.observations[i*self.batch_size+1:(i+1)*self.batch_size+1,
                                                                  self.border:-self.border,
                                                                  self.border:-self.border] / self.median_i
                yield input_validation

    def predict(self):
        print("Predicting velocities with DeepVel...")

        tmp = np.load('network/normalization.npz')
        _, _, min_v, max_v = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']

        start = time.time()
        out = self.model.predict(self.validation_generator(), steps=self.n_frames, verbose=1)
        end = time.time()

        print(f"Prediction took {end - start:.2f} seconds...")

        for i in range(6):
            out[:, :, :, i] = out[:, :, :, i] * (max_v[i] - min_v[i]) + min_v[i]

        out *= 10  # Transform back units

        hdu = fits.PrimaryHDU(out)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(self.output, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepVel prediction')
    parser.add_argument('-o', '--out', help='Output file')
    parser.add_argument('-i', '--in', help='Input file')
    parser.add_argument('-b', '--border', help='Border size in pixels', default=0)
    parsed = vars(parser.parse_args())

    f = fits.open(parsed['in'])
    imgs = f[0].data

    out = deepvel(imgs, parsed['out'], border=int(parsed['border']))
    out.define_network()
    print(out.model.summary())
    out.predict()
