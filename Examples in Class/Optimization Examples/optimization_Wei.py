#!/usr/bin/env python3
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

# I like to show you that we can tune the alpha value to approximate the value of y
# You can test it with y = 0.5, 0.7, 0.9
#y = 0.1
#y = 0.5
#y = 0.7
y = 0.9

eng, q = sf.Engine(1)

alpha = tf.Variable(0.1)
with eng:
    Dgate(alpha) | q[0]
state = eng.run('tf', cutoff_dim=7, eval=False)

# loss is probability for the Fock state n=1
prob = state.fock_prob([0])
loss = tf.square(prob - y) 

# Set up optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

# Create Tensorflow Session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Carry out optimization
for step in range(50):
    prob_val, _ = sess.run([prob, minimize_op])
    print("Value at step {}: {}".format(step, prob_val))
	

# Your exercise is to use this code to build a quantum model to fit this linear dataset:	
# x = [[0.1], [0.4], [0.6], [0.8]]
# y = [[0.2], [0.5], [0.7], [0.9]]	
# Your first step is to write a classical Tensorflow program to approximate the value of y (y=0.1, or 0.5, 0r 0.9)
# then write a classical Tensorflow program to fit this dataset.
