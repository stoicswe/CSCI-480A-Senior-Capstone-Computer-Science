import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

class Policy_net_quantum:
    def __init__(self, name: str, env):

        ob_space = env.observation_space
        act_space = env.action_space
        #input = 4
        #output policy_net = 2
        #ouput value_net = 1
        with tf.variable_scope(name):
            with tf.variable_scope('policy_net'):
                self.P_input = tf.placeholder(dtype=tf.float32, shape=[1,4])
                self.P_eng, self.Pq = sf.Engine(4)
                self.Pbs1 = tf.Variable(0.1)
                self.Pbs2 = tf.Variable(0.1)
                self.Pbs3 = tf.Variable(0.1)
                self.Pbs4 = tf.Variable(0.1)
                self.Pbs5 = tf.Variable(0.1)
                self.Pbs6 = tf.Variable(0.1)
                self.Pd1 = tf.Variable(0.1)
                self.Pd2 = tf.Variable(0.1)
                self.Pd3 = tf.Variable(0.1)
                self.Pd4 = tf.Variable(0.1)
                self.Ps1 = tf.Variable(0.1)
                self.Ps2 = tf.Variable(0.1)
                self.Ps3 = tf.Variable(0.1)
                self.Ps4 = tf.Variable(0.1)
                self.Pv1 = tf.Variable(0.1)
                self.Pv2 = tf.Variable(0.1)
                self.Pv3 = tf.Variable(0.1)
                self.Pv4 = tf.Variable(0.1)
                
                with self.P_eng:
                    Dgate(self.P_input[0][0], 0.)  | self.Pq[0]
                    Dgate(self.P_input[0][1], 0.)  | self.Pq[1]
                    Dgate(self.P_input[0][2], 0.)  | self.Pq[2]
                    Dgate(self.P_input[0][3], 0.)  | self.Pq[3]

                    BSgate(self.Pbs1)           | (self.Pq[0], self.Pq[1])
                    BSgate()                    | (self.Pq[0], self.Pq[1])

                    BSgate(self.Pbs2)           | (self.Pq[0], self.Pq[2])
                    BSgate()                    | (self.Pq[0], self.Pq[2])

                    BSgate(self.Pbs3)           | (self.Pq[0], self.Pq[3])
                    BSgate()                    | (self.Pq[0], self.Pq[3])

                    BSgate(self.Pbs4)           | (self.Pq[1], self.Pq[2])
                    BSgate()                    | (self.Pq[1], self.Pq[2])

                    BSgate(self.Pbs5)           | (self.Pq[1], self.Pq[3])
                    BSgate()                    | (self.Pq[1], self.Pq[3])

                    BSgate(self.Pbs6)           | (self.Pq[2], self.Pq[3])
                    BSgate()                    | (self.Pq[2], self.Pq[3])

                    Dgate(self.Pd1)             | self.Pq[0]
                    Dgate(self.Pd2)             | self.Pq[1]
                    Dgate(self.Pd3)             | self.Pq[2]
                    Dgate(self.Pd4)             | self.Pq[3]

                    Sgate(self.Ps1)             | self.Pq[0]
                    Sgate(self.Ps2)             | self.Pq[1]
                    Sgate(self.Ps3)             | self.Pq[2]
                    Sgate(self.Ps4)             | self.Pq[3]

                    Vgate(self.Pv1)             | self.Pq[0]
                    Vgate(self.Pv2)             | self.Pq[1]
                    Vgate(self.Pv3)             | self.Pq[2]
                    Vgate(self.Pv4)             | self.Pq[3]
                
                self.Pstate = self.P_eng.run('tf', cutoff_dim=10, eval=False)
                self.Pp0 = self.Pstate.fock_prob([0,0,0,2])
                self.Pp1 = self.Pstate.fock_prob([0,0,2,0])
                self.Pp2 = self.Pstate.fock_prob([0,2,0,0])
                self.Pp3 = self.Pstate.fock_prob([2,0,0,0])
                self.Pnormalization = self.Pp0 + self.Pp1 + self.Pp2 + self.Pp3 + 1e-10
                self.Poutput = [self.Pp0/self.Pnormalization, self.Pp1/self.Pnormalization]

                self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
                self.layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                self.layer_2 = tf.layers.dense(inputs=self.layer_1, units=20, activation=tf.tanh)
                self.layer_3 = tf.layers.dense(inputs=self.layer_2, units=2, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=self.layer_3, units=2, activation=tf.nn.softmax)

                self.layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                self.layer_2 = tf.layers.dense(inputs=self.layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=self.layer_2, units=1, activation=None)
                

            with tf.variable_scope('value_net'):
                self.V_input = tf.placeholder(dtype=tf.float32, shape=[1,4])
                self.V_eng, self.Vq = sf.Engine(4)
                self.Vbs1 = tf.Variable(0.1)
                self.Vbs2 = tf.Variable(0.1)
                self.Vbs3 = tf.Variable(0.1)
                self.Vbs4 = tf.Variable(0.1)
                self.Vbs5 = tf.Variable(0.1)
                self.Vbs6 = tf.Variable(0.1)
                self.Vd1 = tf.Variable(0.1)
                self.Vd2 = tf.Variable(0.1)
                self.Vd3 = tf.Variable(0.1)
                self.Vd4 = tf.Variable(0.1)
                self.Vs1 = tf.Variable(0.1)
                self.Vs2 = tf.Variable(0.1)
                self.Vs3 = tf.Variable(0.1)
                self.Vs4 = tf.Variable(0.1)
                self.Vv1 = tf.Variable(0.1)
                self.Vv2 = tf.Variable(0.1)
                self.Vv3 = tf.Variable(0.1)
                self.Vv4 = tf.Variable(0.1)

                with self.V_eng:
                    Dgate(self.V_input[0][0], 0.)  | self.Vq[0]
                    Dgate(self.V_input[0][1], 0.)  | self.Vq[1]
                    Dgate(self.V_input[0][2], 0.)  | self.Vq[2]
                    Dgate(self.V_input[0][3], 0.)  | self.Vq[3]

                    BSgate(self.Vbs1)           | (self.Vq[0], self.Vq[1])
                    BSgate()                    | (self.Vq[0], self.Vq[1])

                    BSgate(self.Vbs2)           | (self.Vq[0], self.Vq[2])
                    BSgate()                    | (self.Vq[0], self.Vq[2])

                    BSgate(self.Vbs3)           | (self.Vq[0], self.Vq[3])
                    BSgate()                    | (self.Vq[0], self.Vq[3])

                    BSgate(self.Vbs4)           | (self.Vq[1], self.Vq[2])
                    BSgate()                    | (self.Vq[1], self.Vq[2])

                    BSgate(self.Vbs5)           | (self.Vq[1], self.Vq[3])
                    BSgate()                    | (self.Vq[1], self.Vq[3])

                    BSgate(self.Vbs6)           | (self.Vq[2], self.Vq[3])
                    BSgate()                    | (self.Vq[2], self.Vq[3])

                    Dgate(self.Vd1)             | self.Vq[0]
                    Dgate(self.Vd2)             | self.Vq[1]
                    Dgate(self.Vd3)             | self.Vq[2]
                    Dgate(self.Vd4)             | self.Vq[3]

                    Sgate(self.Vs1)             | self.Vq[0]
                    Sgate(self.Vs2)             | self.Vq[1]
                    Sgate(self.Vs3)             | self.Vq[2]
                    Sgate(self.Vs4)             | self.Vq[3]

                    Vgate(self.Vv1)             | self.Vq[0]
                    Vgate(self.Vv2)             | self.Vq[1]
                    Vgate(self.Vv3)             | self.Vq[2]
                    Vgate(self.Vv4)             | self.Vq[3]
                
                self.Vstate = self.V_eng.run('tf', cutoff_dim=10, eval=False)
                self.Vp0 = self.Vstate.fock_prob([0,0,0,2])
                self.Vp1 = self.Vstate.fock_prob([0,0,2,0])
                self.Vp2 = self.Vstate.fock_prob([0,2,0,0])
                self.Vp3 = self.Vstate.fock_prob([2,0,0,0])
                self.Vnormalization = self.Vp0 + self.Vp1 + self.Vp2 + self.Vp3 + 1e-10
                self.Voutput = [self.Vp0/self.Vnormalization]
            
            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        act, sto = tf.get_default_session().run([self.Poutput, self.Voutput], feed_dict={self.P_input: obs, self.V_input: obs})
        return np.argmax(act), np.array(sto)
    
    def get_action_prob(self, obs):
        return np.array(tf.get_default_session().run([self.Poutput], feed_dict={self.P_input: obs}))
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        