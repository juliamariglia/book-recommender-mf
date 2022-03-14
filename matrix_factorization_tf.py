# libraries

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics.pairwise import cosine_similarity

# mf class
class MF:

    """
    This class takes in a user-rating matrix and
    uses collaborative filtering via TF embeddings
    to generate item recommendations.
    
    Parameters
    ----------
    rating_matrix: array, m by n array of user ratings (m) by items (n)
    k: int, number of latent features
    learning_rate: float
    epochs: int
    """
    
    def __init__(self, rating_matrix, k = 20, learning_rate = 0.05, epochs = 10, metrics = ['mae', 'mse']):
        self.k = k
        self.rating_matrix = np.array(rating_matrix)
        self.m, self.n = rating_matrix.shape
        self.R_users, self.R_items = rating_matrix.nonzero()
        self.R_ratings = rating_matrix[rating_matrix.nonzero()]
        self.metrics = metrics
        self.lr = learning_rate
        self.epochs = epochs
        
    def create_model(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr)
        
        user_inputs = Input(shape=(1,), name="user")
        item_inputs = Input(shape=(1,), name="item")
        user_embeddings = Embedding(input_dim=self.m, output_dim=self.k, name="user_embedding")(user_inputs)
        item_embeddings = Embedding(input_dim=self.n, output_dim=self.k, name="item_embedding")(item_inputs)

        user_flatten = keras.layers.Flatten(name='FlattenUsers')(user_embeddings)
        item_flatten = keras.layers.Flatten(name='FlattenItems')(item_embeddings)
        
        dot_product = tf.keras.layers.Dot(axes=1, name="dot_product")([user_flatten, item_flatten])
        
        self.model = tf.keras.Model(name="matrix_factorization",
                               inputs=[user_inputs, item_inputs],
                               outputs=dot_product)
        
        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=self.metrics)

        self.checkpoint_path = 'model_checkpoints/checkpoint'
        self.checkpoint = ModelCheckpoint(filepath = self.checkpoint_path, frequency = 'epoch', save_weights_only = True)
        
        return self.model
    
    def train(self):
        """
        Set params for training.
        """
        self.model = self.create_model()
        
        self.history = self.model.fit([self.R_users, self.R_items],
                                      self.R_ratings,
                                      epochs = self.epochs,
                                      callbacks = [self.checkpoint]) #, TODO: early stopping callbacks
        return self.history
        
    def get_sim_matrix(self):

        self.model = self.create_model()
        self.model.load_weights('model_checkpoints/checkpoint')
        self.model.evaluate([self.R_users, self.R_items], self.R_ratings)
        # get predicted Rating matrix
        self.pred_rating_matrix = np.dot(self.model.get_layer(name="user_embedding").weights[0].numpy(),
                                    self.model.get_layer(name="item_embedding").weights[0].numpy().T)
        self.pred_user_embeddings = self.model.get_layer(name="user_embedding").weights[0].numpy()
        self.pred_item_embeddings = self.model.get_layer(name="item_embedding").weights[0].numpy()
        self.p_rm_transpose = np.transpose(self.pred_rating_matrix)
        self.cosine_sim_matrix = cosine_similarity(self.p_rm_transpose)
        
        return self.pred_rating_matrix, self.pred_user_embeddings, self.pred_item_embeddings, self.cosine_sim_matrix
