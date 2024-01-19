import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, ACTMAP

#find sum of future rewards
def find_sum(state, action, V, gamma, env):
    result = 0
    reward, next_state, done = env.step(state, action)
    p = 1 - env.slip
    result += p * (reward + gamma * V[next_state])
    alt_action = ACTMAP[action]
    s_reward, s_next_state, s_done = env.step(state, alt_action)
    p_s = env.slip
    result += p_s * (s_reward + gamma * V[s_next_state])
    return result


def execute_policy(policy, env):
    state = env.reset()
    done = False
    while not done:
        action = int(policy[state])
        reward, next_state, done = env.step(state, action)
        print(f'state: {state}, action: {["UP", "DOWN", "LEFT", "RIGHT"][action]}, reward: {reward}')
        env.plot(state, action)
        state = next_state


env = Maze()


V, Q = np.zeros(env.snum), np.zeros((env.snum, env.anum))


deltas, gamma, theta = [], 0.9, 1.6

# Value Iteration
for iteration in range(1000):
    delta = 0
    for s in range(env.snum):
        v = V[s]
        Q[s, :] = [find_sum(s, a, V, gamma, env) for a in range(env.anum)]
        V[s] = np.max(Q[s, :])
        delta = max(delta, np.abs(v - V[s]))
    deltas.append(delta)
    if delta < theta:
        break




policy = np.argmax(Q, axis=1)


execute_policy(policy, env)

#save my optimal Q
# np.save('ValQOptimal.npy', Q)

