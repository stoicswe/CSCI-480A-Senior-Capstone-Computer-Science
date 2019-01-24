import numpy as np
from chain_links import Chain

n = 10
chain_env = Chain(n, 5, 1, 5)
total_steps = 2**(n-2)

#Initialize table with all zeros
Q = np.zeros([chain_env.get_observation_space(),chain_env.get_action_space()])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
jList = []
rList = []
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
print("Average Steps:")
print(np.mean(jList))