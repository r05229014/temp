"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value

def preprocess(I, down_scale=True, bin_pic=True):
    I = I[35:195]
    if down_scale:
        I = I[::2, ::2]
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y / 255.

    if bin_pic:  # Turn gray scale to binary scale
        y[y >= 0.5] = 1.
        y[y < 0.5] = 0.

    y = y.reshape((6400))

    return np.expand_dims(y.astype(np.float32), axis=0)

class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess
		self.action_size = 3

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .99
		self.tau   = .125

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=4000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.action_size]) # where we will feed de/dC (from critic)
		
		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.global_variables_initializer())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		#state_input = Input(shape=self.env.observation_space.shape)
		state_input = Input(shape=(6400,))
		h1 = Dense(512, activation='relu')(state_input)
		h2 = Dense(256, activation='relu')(h1)
		output = Dense(self.action_size, activation='softmax')(h2)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		#state_input = Input(shape=self.env.observation_space.shape)
		state_input = Input(shape=(6400,))
		state_h1 = Dense(512, activation='relu')(state_input)
		state_h2 = Dense(256)(state_h1)
		
		action_input = Input(shape=(self.action_size,))
		action_h1    = Dense(256)(action_input)
		
		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(64, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
            for sample in samples:
                cur_state, action, reward, new_state, done = sample
                if not done:
                    target_action = self.target_actor_model.predict(new_state)
                    future_reward = self.target_critic_model.predict(
                                [new_state, target_action])[0][0]
                    reward += self.gamma * future_reward
                reward = np.array(reward).reshape((1, 1))
                self.critic_model.fit([cur_state, action], reward, verbose=0)
                #self.critic_model.train_on_batch([cur_state, action], reward)
		
	def train(self):
            batch_size = 64
            if len(self.memory) < batch_size:
                return

            rewards = []
            samples = random.sample(self.memory, batch_size)
            self._train_critic(samples)
            self._train_actor(samples)
            
	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.target_critic_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
            self.epsilon *= self.epsilon_decay
            probs = self.actor_model.predict(cur_state)

            if np.random.random() < self.epsilon:
                #return self.env.action_space.sample()
                action = np.random.choice(self.action_size, 1, p=probs[0])[0]
                return action, probs
    
            return np.argmax(probs), probs
            #return np.random.choice(self.action_size, 1, p=probs[0])[0], probs

def main():
        sess = tf.Session()
        K.set_session(sess)
        env = gym.make("Pong-v0")
        #env = gym.make("Pendulum-v0")
        actor_critic = ActorCritic(env, sess)

        num_trials = 10000
        trial_len  = 500
        prev_state = None
        state_size = [1, 6400]
        total_reward = 0.
        nn = 0

        #action = np.random.choice(3, 1)[0]
        #
        #if action == 0:
        #    action2 = 1
        #elif action == 1:
        #    action2 = 2
        #elif action == 2:
        #    action2 = 3
        #print(cur_state.shape)
        f = open('train_ac.txt', 'w')

        for episode in range(num_trials):
            cur_state = env.reset()
            for _ in range(20):
                prev_state = cur_state 
                cur_state, _, _, _ = env.step(1)

            prev_state = preprocess(prev_state, True, False)
            nn = 1
            while True:
                    cur_state = preprocess(cur_state, True, False)
                    x = cur_state - prev_state if prev_state is not None else np.zeros(state_size)
                    #plt.imshow(x.reshape((80, 80)), cmap='gray')
                    #plt.show()
                    action, probs = actor_critic.act(x)
                    if action == 0:
                        action2 = 1
                    elif action == 1:
                        action2 = 2
                    elif action == 2:
                        action2 = 3
                    
                    new_state, reward, done, _ = env.step(action2)
                    new_state2 = preprocess(new_state)
                    #print(cur_state.shape, probs.shape, reward, new_state2.shape)
                    actor_critic.remember(cur_state, probs, reward, new_state2, done)
                    
                    if nn % 128 == 0: 
                        actor_critic.train()

                    prev_state = cur_state
                    cur_state = new_state
                    total_reward += reward
                    nn = nn + 1
                    if done:
                        state = env.reset()
                        print('Episode:', episode, 'Total reward:', total_reward)
                        f.write(str(episode)+':'+str(total_reward))
                        prev_state = None
                        total_reward = 0
                        if (episode+1) % 50 == 0:
                            actor_critic.update_target()
                        break
        
        actor_critic.actor_model.save('actor.h5')
        actor_critic.critic_model.save('critictor.h5')

if __name__ == "__main__":
        main()
