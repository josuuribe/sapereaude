import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import random
import kernel
import kernel.Interfaces.IStoreManager
import os
import warnings
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


class Neural:
    """
    This class computes and manages the neural network.
    """

    def __init__(self, store_manager: kernel.Interfaces.IStoreManager):
        """
        Constructor that initializes entity
        :param store_manager: Store that manages data to be get or set.
        """
        warnings.filterwarnings('ignore')
        self.__store_manager__ = store_manager
        self.__activation_function__ = tf.nn.relu
        self.__learning_rate__ = kernel.get_learning_rate()
        self.__session__ = tf.Session()
        self.__batches__ = kernel.get_batch_size()
        self.__epochs__ = kernel.get_epochs()
        self.__data_train__ = None
        self.__target_train__ = None
        self.__data_test__ = None
        self.__target_test__ = None
        self.__input_size__ = 0
        self.__output_size__ = 0

    def process(self, data, target):
        """
        This method prepares to data to be used as train and test in training mode.
        :param data: Data to be trained.
        :param target: Target for these data.
        """
        if len(data.shape) == 1:
            self.__input_size__ = 1
            data = data.reshape(1, -1)
        else:
            self.__input_size__ = len(data[0])
        self.__output_size__ = max(target) + 1
        self.__data_train__ = np.nan_to_num(data, 0)
        self.__data_train__ = data
        self.__target_train__ = target
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.__data_train__,
                                                                            self.__target_train__,
                                                                            test_size=0.20,
                                                                            random_state=42)
        scaler = preprocessing.StandardScaler().fit(x_train)
        self.__data_train__ = scaler.transform(x_train)
        self.__target_train__ = y_train
        scaler = preprocessing.StandardScaler().fit(x_test)
        self.__data_test__ = scaler.transform(x_test)
        self.__target_test__ = y_test

    def create_softmax(self):
        """
        Creates softmax architecture for this neural network.
        """
        tf.reset_default_graph()
        self.input_pl = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.__input_size__],
            name="inputplaceholder")
        self.output_pl = tf.placeholder(
            dtype=tf.int16,
            shape=[None, self.__output_size__],
            name="userdefinedoutput")

        dense2 = tf.layers.dense(inputs=self.input_pl,
                                 units=2048,
                                 activation=self.__activation_function__,
                                 name="2_dense_layer")
        dense3 = tf.layers.dense(inputs=dense2,
                                 units=512,
                                 activation=self.__activation_function__,
                                 name="3_dense_layer")
        dense4 = tf.layers.dense(inputs=dense3,
                                 units=64,
                                 activation=self.__activation_function__,
                                 name="4_dense_layer")
        self.network_prediction = tf.layers.dense(inputs=dense4,
                                                  units=self.__output_size__,
                                                  activation=None,
                                                  name="prediction_dense_layer")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.output_pl,
            logits=self.network_prediction)
        self.loss_tensor = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(self.__learning_rate__).minimize(self.loss_tensor)

    def create_rnn(self):
        """
        Demo about how to use a RNN
        """
        rnn_size = 10
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.input_pl = tf.placeholder(tf.float64, [None, self.__input_size__])
        self.output_pl = tf.placeholder(tf.int32, [None, self.__output_size__])

        embedding_matrix = tf.Variable(tf.random_uniform([200, 20], -1.0, 1.0))
        embedding_output = tf.nn.embedding_lookup(embedding_matrix, self.__data_train__)

        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
        output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
        output = tf.nn.dropout(output, self.dropout_keep_prob)

        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[self.__output_size__]))
        logits_out = tf.nn.softmax(tf.add(tf.matmul(last, weight), bias))

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=self.output_pl))

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(self.output_pl, tf.int64)), tf.float32))
        self.optimizer = tf.train.RMSPropOptimizer(self.__learning_rate__).minimize(self.loss)

    def reset_and_train_network_rnn(self,
                                    batch_size):
        """
        Demo about how to train a RNN.
        :param batch_size:
        :return:
        """
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []

        for epoch in range(batch_size):
            shuffle_ix = np.random.permutation(np.arange(len(self.__data_train__)))
            x_train = x_train[shuffle_ix]
            y_train = y_train[shuffle_ix]

            num_baches = int(len(x_train) / batch_size) + 1

            for i in range(num_baches):
                min_ix = i * batch_size
                max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
                x_train_batch = x_train[min_ix:max_ix]
                y_train_batch = y_train[min_ix:max_ix]

                train_dic = {self.input_pl: x_train_batch, self.output_pl: y_train_batch,
                             self.self.dropout_keep_prob: 0.5}
                self.__session__.run(self.optimizer, feed_dict=train_dic)

            temp_train_loss, temp_train_acc = self.__session__.run([self.loss, self.accuracy], feed_dict=train_dic)
            train_loss.append(temp_train_loss)
            train_acc.append((temp_train_acc))

            test_dict = {self.input_pl: self.__data_test__, self.output_pl: self.__target_test__,
                         self.dropout_keep_prob: 1.0}
            temp_test_loss, temp_test_acc = self.__session__.run([self.loss, self.accuracy], feed_dict=test_dict)
            test_loss.append(temp_test_loss)
            test_acc.append(temp_test_acc)
            print("Epoch {} completado, Loss: {:.3f}, Acc: {:.3f}".format(i + 1, temp_test_loss, temp_test_acc))

    def reset_and_train_network(self,
                                verbose=True):
        """
        Demo about how to reset and train a RNN.
        :param verbose: If debugging information about training should be printed.
        :return: Loss history, very useful to see network benchmark.
        """
        self.__session__ = tf.Session()
        init = tf.global_variables_initializer()  # https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
        self.__session__.run(init)

        zipped = list(zip(self.__data_train__, self.__target_train__))

        loss_history = list()
        for _ in range(self.__epochs__):
            datax = list()
            datay = list()
            for _ in range(self.__batches__):
                samp = random.choice(zipped)
                datax.append(samp[0])
                one_hot = [0] * self.__output_size__
                one_hot[samp[1]] = 1
                datay.append(one_hot)
            _, l = self.__session__.run([self.optimizer, self.loss_tensor],
                                        feed_dict={self.input_pl: datax,
                                                   self.output_pl: datay})
            if verbose:
                print(l)
            loss_history.append(l)
        return loss_history

    def predict(self, data):
        """
        Predicts a value with given data.
        :param data: Data used to predict value.
        :return: Predicted value.
        """
        self.__session__ = tf.Session()
        init = tf.global_variables_initializer()  # https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
        self.__session__.run(init)
        predicted_values = self.__session__.run(self.network_prediction,
                                                feed_dict={
                                                    self.input_pl: data})
        return np.argmax(predicted_values, axis=1)

    def evaluate_network(self, data, target):
        """
        Evaluates network, useful to test network efficiency.
        :param data: Data to evaluate.
        :param target: Target value.
        :return: Predicted values and loss for each predicted value.
        """
        datay = list()
        for x in target:
            datasetY2 = [0] * self.__output_size__
            datasetY2[x] = 1.0
            datay.append(datasetY2)
        predicted_values, loss_dataset = self.__session__.run([self.network_prediction, self.loss_tensor],
                                                              feed_dict={
                                                                  self.input_pl: data,
                                                                  self.output_pl: datay})
        return predicted_values, loss_dataset

    def squared_error(self, data, target):
        """
        Computes the squared error, useful to test network fiability.
        :param data: Data to evaluate.
        :param target: Target value.
        :return: Mean squared error.
        """
        datay = list()
        for x in target:
            datasetY2 = [0] * self.__output_size__
            datasetY2[x] = 1.0
            datay.append(datasetY2)
        predicted_values = self.__session__.run(self.network_prediction,
                                                feed_dict={self.input_pl: data})
        mse = mean_squared_error(predicted_values, datay)
        return mse

    def compute_success(self, target, predicted):
        """
        Computes success % based on correct targets and predicted targets.
        :param target: Target value.
        :param predicted: Predicted values.
        :return: Number of correct matches vs wrong between range 0-100, 100 is 100% success.
        """
        ok = 0
        for i in range(len(target)):
            if (np.argmax(predicted[i]) == target[i]):
                ok += 1
        return ok / len(target)

    def visualize_function(self, function, name):
        """
        Visualizes different functions to see performance.
        :param function: Function to evaluate.
        :param name: Function name for print in graph.
        """
        inputdata = np.arange(-5.0, 5.0, 0.1)
        mydatainput = tf.Variable(inputdata)

        functionoutput = function(mydatainput)

        with tf.Session() as temp_session:
            init = tf.global_variables_initializer()
            temp_session.run(init)
            activationdata = functionoutput.eval(session=temp_session)

            plt.plot(inputdata, activationdata)
            plt.xlabel("input")
            plt.ylabel("activation")
            plt.title(name)
            plt.show()

    def save_model(self, user_id):
        """
        Saves the model for this session in user folder.
        :param user_id: User id to get the folder to be used to save.
        """
        root = self.__store_manager__.get_model_folder(user_id)
        f_tf = os.path.join(root, str(user_id) + ".model")
        saver = tf.train.Saver()
        saver.save(self.__session__, f_tf)

    def load_model(self, user_id):
        """
        Loads the model in this user session.
        :param user_id: User id to get the model to be loaded.
        """
        root = self.__store_manager__.get_model_folder(user_id)
        f_tf = os.path.join(root, str(user_id) + ".model.meta")
        saver = tf.train.import_meta_graph(f_tf)
        saver.restore(self.__session__, tf.train.latest_checkpoint(root))
        graph = tf.get_default_graph()
        self.input_pl = graph.get_tensor_by_name("inputplaceholder:0")
        self.network_prediction = graph.get_tensor_by_name("prediction_dense_layer/BiasAdd:0")
