import tensorflow as tf
import train_gan_q_learning as train
import cartpole_networks as networks
import gym

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

def main():
    sess = tf.Session()
    gen = networks.Generator(sess)
    dis = networks.Discriminator(sess)
    dis_copy = networks.Discriminator_copy

    #env = gym.make('CartPole-v0')
    #env = gym.make('FrozenLake-v0')
    env = gym.make('FrozenLakeNotSlippery-v0') #supposibly removes slippery surfaces
    train.learn(env,
                sess,
                100, #1000
                10000, 
                0.99, 
                dis,
                dis_copy,
                gen,
                n_gen=5,
                log_dir='C:/CSCLOGS/')

if __name__ == '__main__' : main()
