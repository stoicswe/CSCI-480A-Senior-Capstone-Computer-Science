import numpy as np
from chain_links import Chain

n = 10
reward_distribution = np.zeros(n)
reward_distribution[0] = 0.1
reward_distribution[n-1] = 1
chain_env = Chain(n, 5, reward_distribution)
total_steps = 2**(n-2)

#Initialize table with all zeros
Q = np.zeros([chain_env.get_observation_space(),chain_env.get_action_space()])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
jList = []
rList = []
print("Should take this number to reach the state: " + str(total_steps))
for i in range(num_episodes):
    s = chain_env.reset()
    rAll = 0
    d = False
    j = 0
    while j < total_steps:
        j+=1
        a = chain_env.sampling()
        s1,r,d = chain_env.step(a)
        rAll += r
        s = s1
        if d == True:
            break
    jList.append(j)
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))
print("Average Steps: " + str(np.mean(jList)))