#from agent_dir.agent import Agent
import tensorflow as tf
import scipy
import gym
import numpy as np
import matplotlib.pyplot as plt


def judge_ball_state(cur_x, rreward, episode):
    cur_x = np.squeeze(cur_x)
    xx = []
    #yy = []
    yyy = set()
    for i in range(160):
        if cur_x[i, 140] >= 0.5 and cur_x[i, 140] <= 0.7:
            yyy.add(i)

    for cc in yyy:
        for sub in range(2):
            if cur_x[cc, 140-sub] >= 0.8:
                if episode <= 100:
                    rreward = rreward + 0.1
    
    return rreward

def preprocess(I, down_scale=True, bin_pic=True):
    I = I[35:195]
    if down_scale:
        I = I[::2, ::2]
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y / 255. 
    
    if bin_pic:  # Turn gray scale to binary scale
        y[y >= 0.5] = 1.
        y[y < 0.5] = 0.
    
    return np.expand_dims(y.astype(np.float32), axis=0)

class Agent_PG():
    #def __init__(self, env, args):
    def __init__(self):
        """
        Initialize every things you need here.
        For example: building your model
        """
    #    super(Agent_PG,self).__init__(env)
        
        self.state_size = [80, 80, 1]
        self.action_size = 3
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.states = []
        self.action_take = []
        self.gradients = []
        self.rewards = []
        self.probs = []

        #if args.test_pg:
            #you can load your model here
        #    print('loading trained model')

    def conv2d(self, x, W, stride=[1, 2, 2, 1]):
        return tf.nn.conv2d(input=x, filter=W, strides=stride, padding='SAME')
    
    def weights(self, name, shape, std_weight=0.1, b_std=0.0):
        name = name.lower()
        b_name = name.replace('w', 'b')

        W_ = tf.get_variable(name, shape, 
        #        initializer=tf.truncated_normal_initializer(stddev=std_weight))
        #        initializer=tf.contrib.keras.initializers.he_uniform())
                initializer=tf.contrib.layers.xavier_initializer())
        B_ = tf.get_variable(b_name, shape[-1], 
                initializer=tf.constant_initializer(b_std))

        return W_, B_

    def _build_model(self):
        print("building model ...")
        self.input = tf.placeholder(tf.float32, [None, 80, 80], name='state_input')
        self.label_inputs = tf.placeholder(tf.float32, [None, self.action_size], name='old_probs')
        self.input_rewards = tf.placeholder(tf.float32, [None, self.action_size], name='rewards')
        f1 = tf.reshape(self.input, [-1, 6400])
        
        f_w2, f_b2 = self.weights('f_w2', [6400, 512])
        f2 = tf.nn.relu(tf.matmul(f1, f_w2))
        
        f_w3, f_b3 = self.weights('f_w3', [512, 256])
        f3 = tf.nn.relu(tf.matmul(f2, f_w3))
        
        f_w4, f_b4 = self.weights('f_w4', [256, 3])
        self.output = tf.nn.softmax(tf.matmul(f3, f_w4))
        
        ## define ppo loss
        #ratio = self.output / (self.label_inputs + 1e-10)
        ratio = tf.exp(self.output - self.label_inputs)
        #ratio = tf.exp(self.label_inputs - self.output)
        pg_losses = -self.input_rewards * ratio
        pg_losses2 = -self.input_rewards * tf.clip_by_value(ratio, 0.8, 1.2)
        #pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        pg_loss = tf.reduce_sum(tf.maximum(pg_losses, pg_losses2))
        
        params = tf.trainable_variables()
        grads = tf.gradients(pg_loss, params)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025)

        self.agent_train = trainer.apply_gradients(grads)

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        #self.gradients.append(np.array(y).astype('float32') - prob)
        #max_prob = prob[np.argmax(prob)]
        #prob[np.argmax(prob)] = prob[action]
        #prob[action] = max_prob
        self.action_take.append(y)
        self.gradients.append(prob)
        self.states.append(state)
        self.rewards.append(reward)
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        
        return discounted_rewards

    def train(self, sess):
        gradients = np.squeeze(np.array(self.gradients))
        action_take = np.array(self.action_take)
        rewards = np.array(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        #rewards = rewards - np.mean(rewards)
        rewards = rewards.repeat(self.action_size).reshape([rewards.shape[0], self.action_size]) * action_take
        
        X = np.squeeze(np.array([self.states]))   # old states
        Y = np.squeeze(np.array([gradients]))     # old probs
        
        ## train
        input_data = {}
        input_data[self.input] = X
        input_data[self.label_inputs] = Y   # old probs
        input_data[self.input_rewards] = rewards  # old rewards

        for i in range(25):
            sess.run(self.agent_train, feed_dict=input_data)
        
        self.states, self.probs, self.gradients, self.action_take, self.rewards = [], [], [], [], []
        

    def make_action(self, sess, observation, test=True, episode=1):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        aprob = sess.run(self.output, feed_dict={self.input: observation})
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        aa = np.random.random()
        if episode <= 250 or aa <= 0.005:
            action = np.random.choice(self.action_size, 1, p=prob[0])[0]
        else:
            action = np.argmax(prob)
        
        return action, prob

if __name__ == "__main__":
    # make a dir
    import os
    dirs = './models'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        os.makedirs(dirs+'/best')
    # start playing environment
    env = gym.make("Pong-v0").unwrapped
    prev_x = None
    score = 0
    save_scores = []

    state_size = [1, 80, 80]
    action_size = 3
    
    # init agent
    agent = Agent_PG()
    agent._build_model()

    saver = tf.train.Saver()
    checkpoint_dir = dirs+'/best'
    nn = 0
    max_last30 = -20
    mean_last30 = 0

    with tf.Session() as sess:
        # init global variables
        sess.run(tf.global_variables_initializer())
        
        #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(3000):
            state = env.reset()
            while True:
                #if episode % 10 == 0:
                #    env.render()
                cur_x = preprocess(state, True, False)
                x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
                prev_x = cur_x
    
                action, prob = agent.make_action(sess, x, episode)
                if(action == 0):
                    action2 = 1
                elif(action == 1):
                    action2 = 2
                elif(action == 2):
                    action2 = 3
                state, reward, done, info = env.step(action2)
                score += reward

                cur_x2 = preprocess(state, False, False) 
                reward = judge_ball_state(cur_x2, reward, episode)  # implement judge ball
                agent.remember(x, action, prob[0], reward)


                if done:
                    agent.train(sess)
                    print('Episode: %d - Score: %f.' % (episode, score))
                    save_scores.append(score)

                    if (len(save_scores)) % 30 == 0:
                        mean_last30 = np.mean(save_scores[-30:])
                        print('Last 30 games mean scores:', mean_last30)
                        
                        with open(dirs+'/training_scores.txt', 'a') as f:
                            for ii,jj in enumerate(save_scores):
                                f.write(str(episode -30 + ii)+':'+str(jj)+'\n')
                        
                        save_scores = []
                        
                        if mean_last30 > max_last30:
                            nn = nn + 1
                            saver.save(sess, dirs+'/best/ppo2_best.ckpt')
                            max_last30 = mean_last30
                        
                    score = 0
                    prev_x = None
                    #env.close()
                    break

            if mean_last30 <= -20.90:
                print('The model break QQ')
                break
        saver.save(sess, dirs+'/policy_ppo2.ckpt')    

