import tensorflow as tf
import math
import os
import numpy as np
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pickle
except:
    import cPickle as pickle


def dense(input_, input_dim, output_dim, activation='linear', name=None):
    if name is None:
        name = 'dense_{}'.format(output_dim)
    with tf.variable_scope(name):
        weights = tf.Variable(
            tf.truncated_normal([input_dim, output_dim],
                                stddev=1.0 / math.sqrt(float(input_dim))),
            name=name+'weights')
        biases = tf.Variable(tf.zeros([output_dim]), name=name+'biases')

        if activation == 'relu' or 'ReLU':
            return tf.nn.relu(tf.matmul(input_, weights) + biases, name='relu')
        elif activation == 'sigmoid':
            return tf.sigmoid(tf.matmul(input_, weights) + biases, name='sigmoid')
        elif activation == 'linear' or 'Linear' or 'LINEAR':
            return tf.matmul(input_, weights) + biases


class Autoencoder:
    def __init__(self,
                 input_dim,
                 model_dir,
                 summary_dir,
                 optimizer=tf.train.AdamOptimizer()):
        self.input_dim = input_dim
        self.model_dir = model_dir
        self.summary_dir = summary_dir
        os.makedirs(model_dir, exist_ok=True)

        with tf.name_scope('input'):
            self.input_ = tf.placeholder(tf.float32, [None, self.input_dim])

        with tf.name_scope('teacher'):
            self.t = tf.placeholder(tf.float32, [None, self.input_dim])

        with tf.variable_scope('layers'):
            self.output = self.build()

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.output - self.t), name='MSE')

        with tf.name_scope('optimizer'):
            self.optimizer = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        os.makedirs(os.path.join(summary_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(summary_dir, 'val'), exist_ok=True)
        # tf.summary.scalar("loss", self.loss)
        # self.summary_op = tf.summary.merge_all()
        with tf.variable_scope('train'):
            self.train_loss = tf.summary.scalar("train_loss", self.loss)
            self.train_writer = tf.summary.FileWriter(
                os.path.join(summary_dir, 'train'), self.sess.graph)
        with tf.variable_scope('val'):
            self.val_loss = tf.summary.scalar("val_loss", self.loss)
            self.val_writer = tf.summary.FileWriter(
                os.path.join(summary_dir, 'val'), self.sess.graph)
        with tf.name_scope('images'):
            self.input_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, 'images/inputs'), self.sess.graph)
            self.output_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, 'images/outputs'), self.sess.graph)

    def build(self):
        x = dense(self.input_, self.input_dim, 512, activation='relu')
        x = dense(x, 512, 256, activation='relu')
        x = dense(x, 256, 32, activation='relu', name='encoded')
        x = dense(x, 32, 256, activation='relu')
        x = dense(x, 256, 512, activation='relu')
        x = dense(x, 512, self.input_dim, activation='sigmoid', name='decoded')
        return x

    def train_on_batch(self, x, y):
        _, loss, summary = self.sess.run([self.optimizer, self.loss, self.train_loss],
                                         feed_dict={self.input_: x, self.t: y})
        return loss, summary

    def fit(self, x, y, batch_size=32, nb_epoch=10, shuffle=False,
            valid_x=None, valid_y=None, valid_step=None, save_step=1):
        assert len(x) == len(y)
        if not(valid_x is None and valid_y is None):
            assert (len(valid_x) == len(valid_y))

        for epoch in range(1, nb_epoch + 1):
            steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
                else len(x) // batch_size + 1
            if shuffle:
                x, y = np.array(zip(*list(np.random.permutation(list(zip(x, y))))))
            for iter_ in range(steps_per_epoch):
                x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
                y_batch = y[iter_ * batch_size: (iter_ + 1) * batch_size]
                train_loss, summary = self.train_on_batch(x_batch, y_batch)
                self.train_writer.add_summary(summary, epoch * steps_per_epoch + iter_)

                end_code = '\n' if steps_per_epoch == iter_ + 1 else '\r'
                print("epoch {} {} / {} loss: {:.5f}".
                      format(epoch, iter_ * batch_size, len(x), train_loss),
                      end=end_code)

            if valid_step is not None and (epoch % valid_step == 0 or epoch == 1):
                val_loss = self._validate(valid_x, valid_y, epoch, batch_size)
                self._visualize(valid_x, epoch)
                print("validation loss: {:.5f}".format(val_loss))

            if epoch % save_step == 0 or epoch == 1:
                self.save("model_epoch_{}".format(epoch))

    def _validate(self, x, y, epoch, batch_size=32):
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        val_loss = 0
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            y_batch = y[iter_ * batch_size: (iter_ + 1) * batch_size]
            l, summary = self.sess.run([self.loss, self.val_loss],
                                       feed_dict={self.input_: x_batch, self.t: y_batch})
            val_loss += l * (len(x_batch) / len(x))
        self.val_writer.add_summary(summary, epoch)
        return val_loss

    def evaluate(self, x, y, batch_size=32):
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        val_loss = 0
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            y_batch = y[iter_ * batch_size: (iter_ + 1) * batch_size]
            l = self.sess.run(self.loss,
                              feed_dict={self.input_: x_batch, self.t: y_batch})
            val_loss += l * (len(x_batch) / len(x))
        return val_loss

    def predict(self, x, batch_size=32):
        outputs = np.empty((0, 784))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.sess.run(self.output,
                              feed_dict={self.input_: x_batch})
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def generate(self, dst_dir, x, batch_size=32):
        os.makedirs(dst_dir, exist_ok=True)
        outputs = self.predict(x, batch_size)

        for index, (i, o) in enumerate(zip(x, outputs)):
            plt.figure(figsize=(4, 2))
            i = np.uint8((i * 255).reshape(28, 28))
            o = np.uint8((i * 255).reshape(28, 28))
            plt.subplot(1, 2, 1)
            plt.imshow(i, cmap='gray')
            plt.xlabel('Input')
            plt.subplot(1, 2, 2)
            plt.imshow(o, cmap='gray')
            plt.xlabel('Output')
            plt.savefig(os.path.join(dst_dir, "{}.png".format(index)))

    def _visualize(self, x, epoch, nb_sample=4):
        _x = x[:nb_sample]
        outputs = tf.cast(self.predict(_x, batch_size=nb_sample), tf.float32)
        with tf.name_scope('epoch_{}'.format(epoch)):
            i = tf.summary.image('inputs',
                                 tf.reshape(_x, [-1, 28, 28, 1]),
                                 nb_sample)
            summary = self.sess.run(i)
            self.input_writer.add_summary(summary, epoch)

            o = tf.summary.image('outputs',
                                 tf.reshape(outputs, [-1, 28, 28, 1]),
                                 nb_sample)
            summary = self.sess.run(o)
            self.output_writer.add_summary(summary, epoch)

    def save(self, file_name):
        return self.saver.save(self.sess, save_path=os.path.join(self.model_dir, file_name))


def data_init(file_path):
    print("Loading MNIST ...    ", end="")
    f = gzip.open(file_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    print("COMPLETE")
    return train_set[0], valid_set[0], test_set[0]


def main():
    file_path = "mnist.pkl.gz"
    train_x, valid_x, test_x = data_init(file_path)
    train_y = train_x.copy()
    valid_y = valid_x.copy()
    input_dim = 784
    nb_epoch = 50
    model_dir = "./params"
    summary_dir = "./logs"

    model = Autoencoder(input_dim, model_dir, summary_dir)
    model.fit(train_x, train_y, save_step=5, nb_epoch=nb_epoch,
              valid_x=valid_x, valid_y=valid_y, valid_step=5)


if __name__ == "__main__":
    main()



