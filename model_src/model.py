import numpy as np
from htm.bindings.algorithms import SpatialPooler, Classifier, TemporalMemory
from htm.bindings.sdr import SDR
from htm.encoders.scalar_encoder import ScalarEncoder
from keras.src.saving.legacy.saved_model.load import metrics

from model_src import default_parameters
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


def get_network():
    net = Sequential()
    net.add(Dense(512, activation='relu'))
    net.add(Dense(64, activation='relu'))
    net.add(Dense(32, activation='relu'))
    net.add(Dense(1, activation='sigmoid'))
    net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return net


class Model:
    def __init__(
            self,
            spatial_pooler_params=default_parameters.spatial_pooler_params,
            temporal_memory_params=default_parameters.temporal_memory_params,
            vel_scalar_encoder_params=default_parameters.velocity_scalar_encoder_params,
            cart_pos_scalar_encoder_params=default_parameters.cart_pos_scalar_encoder_params,
            pole_ang_scalar_encoder_params=default_parameters.pole_ang_scalar_encoder_params,
            # bin_scalar_encoder_params=default_parameters.bin_scalar_encoder_params,
            # horizon_scalar_encoder_params=default_parameters.horizon_scalar_encoder_params
    ):
        self.spatial_pooler = SpatialPooler(**(spatial_pooler_params or {}))
        self.temporal_memory = TemporalMemory(**(temporal_memory_params or {}))
        self.classifier = Classifier()
        self.net_classifier = get_network()
        self.spatial_pooler_input_dimensions = spatial_pooler_params['inputDimensions']
        self.encoding = SDR(spatial_pooler_params['inputDimensions'])
        self.columns = SDR(self.spatial_pooler.getColumnDimensions())

        self.vel_scalar_encoder = ScalarEncoder(vel_scalar_encoder_params)
        self.cart_pos_scalar_encoder = ScalarEncoder(cart_pos_scalar_encoder_params)
        self.pole_ang_scalar_encoder = ScalarEncoder(pole_ang_scalar_encoder_params)

        # self.bin_scalar_encoder = ScalarEncoder(bin_scalar_encoder_params)
        # self.horizon_scalar_encoder = ScalarEncoder(horizon_scalar_encoder_params)

        self.column_dimensions = self.spatial_pooler.getColumnDimensions()[0] * \
                                 self.spatial_pooler.getColumnDimensions()[1]

    def learn(self, examples):
        for episode in tqdm(examples):
            for (cart_pose, cart_vel, pole_ang, pole_ang_vel), action, reward, horizon in episode:
                self._learn_example(cart_pose, cart_vel, pole_ang, pole_ang_vel, action, horizon, reward)

    def predict(self, cart_pose, cart_vel, pole_ang, pole_ang_vel, horizon, reward):
        self.encoding.dense = self._pack_input(cart_pose, cart_vel, pole_ang, pole_ang_vel, reward, horizon)
        self.spatial_pooler.compute(self.encoding, True, self.columns)
        self.temporal_memory.compute(self.columns, learn=True)
        action_to_take = np.argmax(self.classifier.infer(self.temporal_memory.getActiveCells()))
        return action_to_take

    def predict_proba(self, image, location):
        self.encoding.dense = self._pack_input(image, location)
        self.spatial_pooler.compute(self.encoding, False, self.columns)
        return self.classifier.infer(self.columns)[1]

    def _learn_example(self, cart_pose, cart_vel, pole_ang, pole_ang_vel, action, horizon, reward):
        self.encoding.dense = self._pack_input(cart_pose, cart_vel, pole_ang, pole_ang_vel, reward, horizon)
        self.spatial_pooler.compute(self.encoding, True, self.columns)
        self.temporal_memory.compute(self.columns, learn=True)
        self.classifier.learn(self.temporal_memory.getActiveCells(), int(action))

    def _pack_input(self, cart_pose, cart_vel, pole_ang, pole_ang_vel, reward, horizon):
        cart_pose_encoding = self.cart_pos_scalar_encoder.encode(cart_pose).dense

        pole_ang_encoding = self.pole_ang_scalar_encoder.encode(pole_ang).dense

        cart_vel_encoding = self.vel_scalar_encoder.encode(cart_vel).dense
        pole_ang_vel_encoding = self.vel_scalar_encoder.encode(pole_ang_vel).dense

        # action_encoding = self.bin_scalar_encoder.encode(reward).dense
        # horizon_encoding = self.horizon_scalar_encoder.encode(horizon).dense
        # concatenated = np.concatenate((cart_pose_encoding, cart_vel_encoding, pole_ang_encoding, pole_ang_vel_encoding,
        #                                action_encoding, horizon_encoding), axis=0)
        concatenated = np.concatenate((cart_pose_encoding, cart_vel_encoding, pole_ang_encoding, pole_ang_vel_encoding),
                                      axis=0)
        return concatenated.reshape(self.spatial_pooler_input_dimensions)

    def train_net(self, examples):
        train = []
        labels = []
        for episode in examples:
            for (cart_pose, cart_vel, pole_ang, pole_ang_vel), action, reward, horizon in episode:
                example_active_cells = self._learn_example_net(cart_pose, cart_vel, pole_ang, pole_ang_vel)
                train.append(example_active_cells.dense.flatten())
                labels.append(action)
        train = np.concatenate([train], axis=0)
        labels = np.asarray(labels, dtype=np.float32)
        df = pd.DataFrame(np.concatenate([train, labels.reshape(-1, 1)], axis=1))
        df.columns = [str(x) for x in range(df.shape[1])]
        self.net_classifier.fit(x=train, y=labels, verbose=1, batch_size=32, epochs=10)

    def _learn_example_net(self, cart_pose, cart_vel, pole_ang, pole_ang_vel):
        self.encoding.dense = self._pack_input(cart_pose, cart_vel, pole_ang, pole_ang_vel, '', '')
        self.spatial_pooler.compute(self.encoding, False, self.columns)
        self.temporal_memory.compute(self.columns, learn=False)
        return self.temporal_memory.getActiveCells()

    def predict_net(self, cart_pose, cart_vel, pole_ang, pole_ang_vel, horizon, reward):
        self.encoding.dense = self._pack_input(cart_pose, cart_vel, pole_ang, pole_ang_vel, horizon, reward)
        self.spatial_pooler.compute(self.encoding, True, self.columns)
        self.temporal_memory.compute(self.columns, learn=True)
        return int(self.net_classifier.predict(self.temporal_memory.getActiveCells().dense.flatten().reshape(1, -1), verbose=0)[0][0] > 0.5)

    def predict_proba_net(self, image, location):
        self.encoding.dense = self._pack_input(image, location)
        self.spatial_pooler.compute(self.encoding, False, self.columns)
        # unpaded = self.columns.sparse.reshape(1, self.columns.sparse.shape[0])
        # padded = np.pad(unpaded.reshape(unpaded.shape[1]), (0, 150 - unpaded.shape[1]), 'constant').reshape(1, 150)
        unpadded = self.columns.dense.flatten()
        padded = np.pad(unpadded, (0, self.column_dimensions - unpadded.shape[0]), 'constant') \
            .reshape(1, self.column_dimensions)
        return self.classifier.predict(padded)
