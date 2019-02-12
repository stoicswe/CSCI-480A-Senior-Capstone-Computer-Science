import tensorflow as tf

y = 0.9
alpha = tf.Variable(0.1)
loss = tf.square(alpha-y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

# Create Tensorflow Session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Carry out optimization
for step in range(50):
    prob_val, _ = sess.run([alpha, minimize_op])
    print("Value at step {}: {}".format(step, prob_val))