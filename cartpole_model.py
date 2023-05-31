# %%
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from helper import plot
# %%
env = gym.make('CartPole-v1')
obs = env.reset()

model = keras.Sequential([
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# %%
def discount_reward(reward, discount_rate=.99):
    discounted = np.array(reward)
    for i in range(len(reward)-2, -1,-1):
        discounted[i] += discounted[i+1] * discount_rate
    
    return discounted

def discount_and_normalize_rewards(rewards, discount_rate=.99):
    discounted = [discount_reward(reward) for reward in rewards]

    flatten_discounted = np.concatenate(discounted, axis=0)

    mean = flatten_discounted.mean()
    std = flatten_discounted.std()

    return [(discounted_reward-mean)/std
            for discounted_reward in discounted]

# %%
def play_one_step(env, obs, model=model, loss_fn=keras.losses.binary_crossentropy):
    with tf.GradientTape() as tape:
        left_prob = model(obs[np.newaxis,...])
        target = tf.constant([[0.0]]) if np.random.random() > left_prob else tf.constant([[1.0]])
        
        loss = loss_fn(target, left_prob)
    grads = tape.gradient(loss, model.trainable_variables)

    obs, reward, done, _ = env.step(int(target.numpy()))

    return obs, reward, done, grads

def play_multiple_step(n_episodes, n_max_step, env=env, model=model, loss_fn=keras.losses.binary_crossentropy):
    all_rewards = []
    all_grads = []
    all_survied = []

    for episode in range(n_episodes):
        obs = env.reset()

        current_rewards = []
        current_grads = []

        for step in range(n_max_step):
            obs, reward, done, grads = play_one_step(env, obs)

            current_rewards.append(reward)
            current_grads.append(grads)

            if done:
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
        all_survied.append(step)

    
    return all_rewards, all_grads, all_survied 

# %%
n_iters = 150
n_episodes_per_update = 10
n_max_steps = 200
gamma = 0.99
optimizer = keras.optimizers.Adam(.01)
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iters):
    all_rewards, all_grads, survived_per_episode = play_multiple_step(
        n_episodes_per_update,n_max_steps
    )

    max_survive_index = np.argmax(survived_per_episode)
    plot(max(survived_per_episode))

    all_final_rewards = discount_and_normalize_rewards(all_rewards, gamma)

    all_mean_grads = []

    for var_index in range(len(model.trainable_variables)):
        mean_grads = []
        
        for episode_index, final_rewards in enumerate(all_final_rewards):
            for step, final_reward in enumerate(final_rewards):
                mean_grads.append(final_reward * all_grads[episode_index][step][var_index])
        
        mean_grads = tf.reduce_mean(mean_grads, axis=0)

        all_mean_grads.append(mean_grads)
      
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

