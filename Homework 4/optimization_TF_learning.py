import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(0.1)
b = tf.Variable(0.1)
output = w*x+b
loss = tf.square(w*x+b-y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_vals = [[0.1], [0.4], [0.6], [0.8]]
y_vals = [[0.2], [0.5], [0.7], [0.9]]

iterations = 100
for step in range(iterations):
    print("Step: {0} ======================================".format(step))
    for i in range(len(x_vals)):
        sess.run([w, minimize_op], feed_dict={x: x_vals[i], y: y_vals[i]})
        out = sess.run([output], feed_dict={x: x_vals[i]})
        print("Input: {0} | Prediction: {2} | Exact Output: {1}".format(x_vals[i], y_vals[i], out[0]))
