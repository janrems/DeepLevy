#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time, datetime
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow_probability as tfp
from tensorflow.python.training import moving_averages

start_time = time.time()
tf.get_default_graph()

name = 'optimiser_control'
d = 1
batch_size = 30
n_maxstep = 2000
n_displaystep = 100
n_neuron = [d, d + 2, d + 2, d]  # d je dimenzija inputa, torej mamo kle situacijo ko je 1,3,3,1 NN
print(batch_size)
G = 10
T = 1  ##temps d'exercice (en année)
dt = 0.005  #
sqrdt = np.sqrt(dt)
b = tf.ones(shape=[d, d], dtype=tf.float64, name='MatrixOfOnes')
b1 = 6  # b*6 + np.random.randn(d,d)
b2 = 0.5  # b*0.5+np.random.randn(d,d)
delta = 0.1
t1 = np.arange(0, delta, dt)
N_delta = len(t1)
#t2 = np.arange(delta, T, dt)
t2 = t1
N_t2 = len(t2)
t = np.concatenate((t1, t2))  # time length
N_time = len(t)
sigma = 0.5  # coefficient de diffusion (la volatilité du marché)
K = 0.5  # prix díexercice
loss0 = 0
print(N_time)
_extra_train_ops = []


# forward propagation algorithm with ReLU as the activation function through Tensorflow
def _one_layer(input_, output_size, activation_fn=tf.nn.relu, stddev=5.0,
               name='linear'):  # kle mas activation function. tole je funkcija za en nivo
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float64,
                            tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size)))
        hidden = tf.matmul(input_, w)
        # shape1 = hidden.get_shape().as_list ()
        hidden_bn = _batch_norm(hidden, name='normal')  # toe je pomoje normiran rezultat nevrona
        if activation_fn != None:  # kle se gor aktivacijska funkcija spusti
            return activation_fn(hidden_bn)
        else:
            return hidden_bn


def _one_time_net(x, name):  # kle je del mreže za en čas
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        x_norm = _batch_norm(x, name='layer0_normal')
        layer1 = _one_layer(x_norm, n_neuron[1], name='layer1')
        layer2 = _one_layer(layer1, n_neuron[2], name='layer2')
        z = _one_layer(layer2, n_neuron[3], activation_fn=None, name='final')  # output nima aktivacijske funkcije
    return z


def _batch_norm(x, name):  # normalizacija za input in pa za outpute na skritih layerjih
    """ Batch normalization """
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable('beta', params_shape, tf.float64,
                               initializer=tf.random_normal_initializer(0.0, stddev=0.1, dtype=tf.float64))
        # tf.random_normal returns a tensor of the specified shape filled with random normal value
        gamma = tf.get_variable('gamma', params_shape, tf.float64,
                                initializer=tf.random_uniform_initializer(0.1, 0.5, dtype=tf.float64))
        moving_mean = tf.get_variable('moving_mean', params_shape, tf.float64,
                                      initializer=tf.constant_initializer(0.0, tf.float64), trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape, tf.float64,
                                          initializer=tf.constant_initializer(1.0, tf.float64), trainable=False)
        mean, variance = tf.nn.moments(x, [0], name='moments')
        _extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.99))
        _extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.99))
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
        y.set_shape(x.get_shape())
    return y


# tole je loop
with tf.Session() as sess:
    dW = tf.random_normal(shape=[N_t2,batch_size, d], stddev=1, dtype=tf.float64)  # 30x1 st.nor.
    X0 = tf.Variable(tf.random_uniform([N_delta, batch_size, d],
                                       minval=-1, maxval=1, dtype=tf.float64), name='X0')  # X deltax30x1, to se uci
    XI = tf.Variable(tf.random_uniform([N_t2, batch_size, d],
                                       minval=-1, maxval=1, dtype=tf.float64), trainable=False,
                     name='XI')  # X od delta daljex30x1 tega se ne uci
    f1 = tf.Variable(tf.random_uniform([d], minval=-1.2, maxval=-0.5, dtype=tf.float64), name='f1')

    allones = tf.ones(shape=[batch_size, d], dtype=tf.float64,
                      name='MatrixOfOnes')
    f0 = allones * f1
    with tf.variable_scope('forward', reuse=tf.compat.v1.AUTO_REUSE):
        for i in range(0, N_delta-1):
            X0[i+1, :, :].assign(
                _one_time_net(XI[i, :, :], str(i + 1) + "X0"))  # kle je mal cudno da i-to mesto updateas z
        X = tf.concat([X0, XI], 0)
        for i in range(0, N_t2):
            dr2 = b2 * (tf.reduce_mean(X[N_delta + i - 1, :, :], axis=0)) * dt
            for j in range(0, batch_size):
                XI[i, j, :].assign(b1 * X[i, j, :] * dt + dr2 + sqrdt * sigma * dW[i,j, :])
            #dW = tf.random_normal(shape=[batch_size, d], stddev=1, dtype=tf.float64)
        X = tf.concat([X0, XI], 0)
        integration1 = 0.5 * tf.reduce_sum(tf.multiply(X[0, :, :], X[0, :, :]), 1)  # ???????????????
        integration2 = tfp.math.trapz(0.5 * tf.reduce_sum(tf.multiply(X[0:N_delta, :, :], X[0:N_delta, :, :]), 2),
                                      axis=0)
        integration3 = tfp.math.trapz(
            tf.reduce_mean(tf.reduce_sum(tf.multiply(X[N_delta:N_time, :, :], X[N_delta:N_time, :, :]),
                                         2), axis=1), axis=0)
        # Cost function
        #J = integration2 + integration1 + K * tf.reduce_sum(tf.multiply(X[-1, :, :] - G, X[-1, :, :] - G), 1)
        J = - tf.math.log(tf.norm(XI[-1,:,:], axis= 1))
        loss = tf.reduce_mean(J)

        # training operations
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                      dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(1.0, global_step, decay_steps=200, decay_rate=0.5, staircase=True)
        # Returns all variables created with trainable=True
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_variables)
        # Define an optimizer using tensorflow
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        train_ops = [apply_op] + _extra_train_ops
        train_op = tf.group(*train_ops)
    with tf.control_dependencies([train_op]):
        train_op_2 = tf.identity(loss, name='train_op2')
    # to save history
    learning_rates = []
    losses = []
    running_time = []
    steps = []
    sess.run(tf.global_variables_initializer())
    i = 0
    loss2 = 10000000
    diff = np.abs(loss2 - loss0)
    try:
        while (diff > 0.000003):  # dokler je sprememba v izgubi večja kot tok
            step = sess.run(global_step)
            XI = sess.run(X)
            steps.append(step)
            currentLoss, currentLearningRate = sess.run([train_op_2, learning_rate])
            losses.append(currentLoss)
            learning_rates.append(currentLearningRate)
            if (i == 0):
                loss0 = 0
            else:
                loss0 = losses[i - 1]
            loss2 = losses[i]
            diff = np.abs(loss2 - loss0)
            i = i + 1
            running_time.append(time.time() - start_time)
            if step % n_displaystep == 0:
                print("step :", step,
                      "loss :", currentLoss,
                      "learning rate :", currentLearningRate)
                # merged = tf.summary.merge_all()
                # writer = tf.summary.FileWriter("maroua",sess.graph)
        Xl = np.array(X.eval())
        end_time = time.time()
        # print ( "running time :" , end_time - start_time )
        # print("\nintegr1 :",sess.run(integration1))
        # print("\nintegr2 :",sess.run(integration2))
        # print("\nintegr3 :",sess.run(integration3))
        # print("\nJ :", sess.run(J))
        # print("\nloss1 :",sess.run(loss1))
    except KeyboardInterrupt:
        print("manually disengaged")
print("\n,N_delta:", N_delta)
print("\n,iii:", i)
print("\n,diff:", diff)
print("\n,X0:", Xl[0:N_delta, 1, 0])
for j in range(10):
    plt.plot(t1[0:-1], Xl[1:N_delta, j, 0])
plt.grid(True)
plt.figure()
plt.plot(t1[0:-1], Xl[1:N_delta, 1, 0])
plt.title("dX")
plt.grid(True)
plt.figure()
n = np.arange(i)
plt.plot(n, losses)
plt.grid(True)
plt.show()

####### TO JE MOJE
plt.plot(t, Xl[:, 10, 0])
plt.show()

# In[ ]:


import time, datetime
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow_probability as tfp
from tensorflow.python.training import moving_averages

start_time = time.time()
tf.get_default_graph()
d = 4
b = tf.ones(shape=[d, d], dtype=tf.float64, name='MatrixOfOnes')
b1 = b * 6 + np.random.randn(d, d)
b2 = b * 0.5 + np.random.randn(d, d)
f1 = tf.Variable(tf.random_uniform([d], minval=-1.2, maxval=-0.5, dtype=tf.float64), name='f1')

allones = tf.ones(shape=[30, d], dtype=tf.float64, name='MatrixOfOnes')
f0 = allones * -1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(f0.eval())

