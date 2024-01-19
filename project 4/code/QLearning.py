import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from evaluation import get_action_egreedy, evaluation
from maze import Maze

def q_learning(env, num_episodes, learning_rate, epsilon, Q_optimal, gamma=0.9):
    # initialize Q
    Q = np.zeros([env.snum, env.anum])

    
    errors = []
    eval_steps, eval_rewards = [], []

    for episode in range(num_episodes):
        
        state = env.reset()

        for step in range(5000):
            # choose action based on e-greedy policy
            action = get_action_egreedy(Q[state], epsilon)

            
            reward, next_state, done = env.step(state, action)

            # update Q value
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)
            Q[state, action] = new_value

            # calculate RMSE
            error = mean_squared_error(Q, Q_optimal, squared=False)
            errors.append(error)

            
            state = next_state

            
            if done:
                break

        # evaluate every 50 episodes
        if episode % 50 == 0:
            avg_step, avg_reward = evaluation(env, Q)
            eval_steps.append(avg_step)
            eval_rewards.append(avg_reward)

    return Q, errors, eval_steps, eval_rewards



maze = Maze()


num_episodes = 5000
learning_rates = [0.1, 0.2, 0.3]
epsilon = 0.1

# Load optimal Q-table
Q_optimal = np.load('ValQOptimal.npy')


fig_error, ax_error = plt.subplots(figsize=(10, 5))
fig_steps, ax_steps = plt.subplots()
fig_rewards, ax_rewards = plt.subplots()

# run Q learning
for learning_rate in learning_rates:
    Q, errors, eval_steps, eval_rewards = q_learning(maze, num_episodes, learning_rate, epsilon, Q_optimal)

    # plot Qerr
    ax_error.plot(errors, label=f'Learning Rate = {learning_rate}')

    # plot evaluation metrics
    ax_steps.plot(np.arange(0, num_episodes, 50), eval_steps, label=f'Learning Rate = {learning_rate}')
    ax_rewards.plot(np.arange(0, num_episodes, 50), eval_rewards, label=f'Learning Rate = {learning_rate}')


ax_error.set_title('RMSE')
ax_error.set_xlabel('Steps')
ax_error.set_ylabel('RMSE')
ax_error.legend()

ax_steps.set_title('Evaluation Steps')
ax_steps.set_xlabel('Episodes')
ax_steps.set_ylabel('Average Steps')
ax_steps.legend()

ax_rewards.set_title('Evaluation Rewards')
ax_rewards.set_xlabel('Episodes')
ax_rewards.set_ylabel('Average Rewards')
ax_rewards.legend()


plt.show()