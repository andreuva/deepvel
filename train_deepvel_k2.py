import h5py
import os
import json
import argparse
from contextlib import redirect_stdout

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Add
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class LossHistory(Callback):
    def __init__(self, root, losses):
        self.root = root        
        self.losses = losses

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs)
        with open(f"{self.root}_loss.json", 'w') as f:
            json.dump(self.losses, f)

    def finalize(self):
        pass

class train_deepvel(object):

    def __init__(self, root, noise, option):
        # GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except:
                pass

        self.root = root
        self.option = option

        self.n_filters = 64
        self.kernel_size = 3        
        self.batch_size = 32
        self.n_conv_layers = 20
        
        self.input_file_images_training = " "
        self.input_file_velocity_training = " "

        self.input_file_images_validation = " "
        self.input_file_velocity_validation = " "

        with h5py.File(self.input_file_images_training, 'r') as f:
            self.n_training_orig, self.nx, self.ny, self.n_times = f["intensity"].shape

        with h5py.File(self.input_file_images_validation, 'r') as f:
            self.n_validation_orig, _, _, _ = f["intensity"].shape
        
        self.batchs_per_epoch_training = self.n_training_orig // self.batch_size
        self.batchs_per_epoch_validation = self.n_validation_orig // self.batch_size

        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print(f"Original training set size: {self.n_training_orig}")
        print(f"   - Final training set size: {self.n_training}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Batches per epoch: {self.batchs_per_epoch_training}")

        print(f"Original validation set size: {self.n_validation_orig}")
        print(f"   - Final validation set size: {self.n_validation}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Batches per epoch: {self.batchs_per_epoch_validation}")

    def training_generator(self):
        with h5py.File(self.input_file_images_training, 'r') as f_images, \
             h5py.File(self.input_file_velocity_training, 'r') as f_velocity:

            images = f_images["intensity"]
            velocity = f_velocity["velocity"]

            while True:
                for i in range(self.batchs_per_epoch_training):
                    input_train = images[i*self.batch_size:(i+1)*self.batch_size].astype('float32')
                    output_train = velocity[i*self.batch_size:(i+1)*self.batch_size].astype('float32')
                    yield input_train, output_train

    def validation_generator(self):
        with h5py.File(self.input_file_images_validation, 'r') as f_images, \
             h5py.File(self.input_file_velocity_validation, 'r') as f_velocity:

            images = f_images["intensity"]
            velocity = f_velocity["velocity"]

            while True:
                for i in range(self.batchs_per_epoch_validation):
                    input_val = images[i*self.batch_size:(i+1)*self.batch_size].astype('float32')
                    output_val = velocity[i*self.batch_size:(i+1)*self.batch_size].astype('float32')
                    yield input_val, output_val

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

        # Save model architecture
        with open(f'{self.root}_model.json', 'w') as f:
            f.write(self.model.to_json())

        with open(f'{self.root}_summary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        plot_model(self.model, to_file=f'{self.root}_model.png', show_shapes=True)

    def compile_network(self):
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))

    def read_network(self):
        print("Reading previous network...")
        with open(f'{self.root}_model.json', 'r') as f:
            json_string = f.read()

        self.model = model_from_json(json_string)
        self.model.load_weights(f"{self.root}_weights.hdf5")

    def train(self, n_iterations):
        print("Training network...")

        # Load previous loss history if continuing
        if self.option == 'continue':
            with open(f"{self.root}_loss.json", 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath=f"{self.root}_weights.hdf5", verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, losses)

        self.model.fit(
            self.training_generator(),
            steps_per_epoch=self.batchs_per_epoch_training,
            epochs=n_iterations,
            callbacks=[self.checkpointer, self.history],
            validation_data=self.validation_generator(),
            validation_steps=self.batchs_per_epoch_validation
        )

        self.history.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepVel')
    parser.add_argument('-o', '--out', help='Output files')
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n', '--noise', help='Noise to add during training', default=0.0)
    parser.add_argument('-a', '--action', help='Action', choices=['start', 'continue'], required=True)
    parsed = vars(parser.parse_args())

    root = parsed['out']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = parsed['noise']

    out = train_deepvel(root, noise, option)

    if option == 'start':           
        out.define_network()        
        
    if option == 'continue':
        out.read_network()

    if option in ['start', 'continue']:
        out.compile_network()
        out.train(nEpochs)
