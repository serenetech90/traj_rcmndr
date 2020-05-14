import tensorflow as tf

class gsk_lstm_cell():
    def __init__(self, in_features, obs_len, num_nodes, lambda_reg):
        self.out_size = tf.placeholder_with_default(input=num_nodes, shape=[], name='out_size')
        self.lambda_reg = tf.Variable(lambda_reg, dtype=tf.float64)
        self.init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64)

        self.outputs = tf.placeholder_with_default(
                                        input=tf.random.normal(shape=[int(in_features.shape[0]),
                                        int(in_features.shape[0])],mean=0, stddev=1, seed=0, dtype=tf.float64),
                                        shape=[int(in_features.shape[0]),
                                              int(in_features.shape[0])], name="outputs")

        self.ngh = tf.placeholder_with_default(
            input=tf.random.normal(shape=[int(in_features.shape[0]), 8],
                                   mean=0, stddev=1, seed=0, dtype=tf.float64),
            shape=[int(in_features.shape[0]), 8], name="ngh")


        with tf.variable_scope("krnl_weights"):
            self.weight_v = tf.Variable(name='weight_v', initial_value= \
                                        self.init_w(shape=(8, int(in_features.shape[0]))),
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

            self.bias_v = tf.Variable(name='bias_v', initial_value= \
                                      self.init_w(shape=(int(in_features.shape[0]),)),
                                      # shape=tf.shape(1,in_features.shape[1].value),
                                      dtype=tf.float64)

            self.weight_o = tf.Variable(name='weight_o', initial_value= \
                                        self.init_w(shape=(int(in_features.shape[0]), num_nodes)),
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

            self.weight_c = tf.Variable(name='weight_c', initial_value= \
                                        self.init_w(shape=(24, obs_len)),
                                        # shape=tf.shape(1,in_features.shape[1].value),
                                        dtype=tf.float64)

        # self.visual_path = tf.placeholder_with_default(input=tf.random.normal(shape=[1, int(in_features.shape[0])],
        #                                                                       mean=0, stddev=1, seed=0,
        #                                                                       dtype=tf.float64),  # dtype=tf.float64,
        #                                                shape=[1, in_features.shape[0]], name="visual_path")

        self.pred_path_band = tf.placeholder_with_default(input=tf.random.normal(shape=[2, 12, num_nodes],
                                                                                 mean=0, stddev=1, seed=0,
                                                                                 dtype=tf.float64),  # dtype=tf.float64,
                                                          shape=[2, 12, num_nodes], name="pred_path_band")

        embedded_spatial_vislet = tf.Variable(tf.matmul(self.weight_v, self.outputs) + self.bias_v)  # 12x10
        # ngh_temp = tf.Variable((self.lambda_reg * self.ngh) )# 12x10
        self.ngh = tf.Variable(tf.multiply(self.ngh, tf.transpose(embedded_spatial_vislet)))
        _, self.cost = tf.gradients(ys=self.ngh, xs=[embedded_spatial_vislet, self.ngh],
                                    stop_gradients= embedded_spatial_vislet,
                                    unconnected_gradients='zero')

        self.cost = tf.squeeze(self.cost)
        self.cost = tf.Variable(tf.nn.relu(self.cost))

        self.temp_path = tf.Variable(tf.matmul(self.weight_c, tf.transpose(self.cost)))  # 24x10
        self.temp_path = tf.Variable(tf.matmul(self.temp_path, self.weight_o))  # 24xn

        self.pred_path_band = tf.reshape(self.temp_path, (2, 12, self.out_size))  # 2x12xn
