import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner

# Define the variational circuit and its output.
X = tf.placeholder(tf.float32, shape=[2])
y = tf.placeholder(tf.float32)
phi = tf.Variable(2.)

eng, q = sf.Engine(2)

with eng:
    # Note that we are feeding 1-d tensors into gates, not scalars!
    Dgate(X[0], 0.) | q[0]
    Dgate(X[1], 0.) | q[1]
    BSgate(phi) | (q[0], q[1])
    BSgate() | (q[0], q[1])

# We have to tell the engine how big the batches (first dim of X) are
# which we feed into gates
num_inputs = X.get_shape().as_list()[0]
state = eng.run('tf', cutoff_dim=10, eval=False)

# Define the output as the probability of measuring |0,2> as opposed to |2,0>
p0 = state.fock_prob([0, 2])
p1 = state.fock_prob([2, 0])
normalization = p0 + p1 + 1e-10
circuit_output = p1 / normalization

loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = tf.round(circuit_output)

# Generate some data
X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
Y_train = [1, 1, 0, 0]
X_test = [[0.25, 0.5], [0.5, 0.25]]
Y_test = [1, 0]
X_pred = [[0.4, 0.5], [0.5, 0.4]]

steps = 100

for i in range(steps):
    if i % 10 == 0:
        print("Epoch {0}, Loss {1}".format(i, sess.run([loss], feed_dict={X: X_train[0], y: Y_train[0]})[0]))
    for j in range(len(Y_train)):
        l = sess.run([minimize_op], feed_dict={X: X_train[j], y: Y_train[j]})
    

print("X       Prediction       Label")
for i in range(len(Y_test)):
    print("{0} || {1} || {2}".format(X_test[i], sess.run(output, feed_dict={X: X_test[i]}), Y_test[i]))