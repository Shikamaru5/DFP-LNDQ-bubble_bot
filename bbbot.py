#https://www.retrogames.cz/play_216-NES.php?language=EN
#This is the link to the website w/ the bubble bobble game.

import os
import time
import pyautogui, sys
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
from sklearn import preprocessing
import cv2
from PIL import ImageGrab
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "F:\Program Files\Tesseract-OCR\Tesseract.exe"

LOAD_MODEL = None
#None
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16_000)])
  except RuntimeError as e:
    print(e)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 128 #Max sample memory.
MIN_REPLAY_MEMORY_SIZE = 72
MINIBATCH_SIZE = 32
UPDATE_TARGET_EVERY = 1
MIN_REWARD = -30  # For model save
#MEMORY_FRACTION = 0.20
timesteps = [1,2,4,8,16,32] #number of timesteps to be future predicted.

# Environment settings
EPISODES = 1_600

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 5  # episodes
MODEL_NAME = "bbbot"

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

class Environment():

    ACTION_SPACE_SIZE = 6
    OBSERVATION_SPACE_VALUES = (80, 80, 3)
    RETURN_IMAGES = True
    #REWARD_SPACE_SIZE = 3
    GOOD = 1
    BAD = 10

    def startGame(self):
        #start the game, giving the user a few seconds to click on the chrome tab after starting the code
        for i in reversed(range(3)):
            print("agent starting in ", i)
            time.sleep(1)

    def action(self, choice):

        
        if choice == 0:
            self.left()

        if choice == 1:
            self.right()

        if choice == 2:
            self.up()

        if choice == 3:
            self.bubble()

        if choice == 4:
            self.left1()

        if choice == 5:
            self.right1()

    def left(self):

        pyautogui.keyDown('left')
        time.sleep(0.5)
        pyautogui.keyUp('left')
        print('left')

    def right(self):

        pyautogui.keyDown('right')
        time.sleep(0.5)
        pyautogui.keyUp('right')
        print('right')

    def left1(self):

        pyautogui.keyDown('left')
        time.sleep(0.25)
        pyautogui.keyUp('left')
        print('left')

    def right1(self):

        pyautogui.keyDown('right')
        time.sleep(0.25)
        pyautogui.keyUp('right')
        print('right')
        
    def up(self):

        pyautogui.keyDown('a')
        pyautogui.keyUp('a')
        print('a')
        
    def bubble(self):

        pyautogui.keyDown('space')
        pyautogui.keyUp('space')
        print('space')

    def reset(self):

        #This is after done is determined and it restarts the game
        #so that you don't have to manually do it after every episode
        #which would interrupt the flow.

        pyautogui.keyDown('enter')
        print('enter')
        time.sleep(0.25)
        pyautogui.keyUp('enter')
        pyautogui.keyDown('enter')
        print('enter')
        pyautogui.keyUp('enter')
        time.sleep(10)

        if self.RETURN_IMAGES:
            img = np.array(self.eyes())

        state = np.array(img)

        self.episode_step = 0

        return state

    def step(self, action):

        self.episode_step += 1

        if self.RETURN_IMAGES:
            img = np.array(self.eyes())

        state = np.array(img)

        screen2 = np.array(ImageGrab.grab(bbox=(650, 340, 800, 365)))
        grey2 = cv2.cvtColor(screen2, cv2.COLOR_BGR2GRAY)
        score1 = pytesseract.image_to_string(grey2)
        if score1.endswith('\n'):
            score1[:-2]

        self.action(action)

        if self.RETURN_IMAGES:
            next_img = np.array(self.eyes())

        new_state = np.array(next_img)

        done = False

        screen3 = np.array(ImageGrab.grab(bbox=(650, 340, 800, 365)))
        grey3 = cv2.cvtColor(screen3, cv2.COLOR_BGR2GRAY)
        score2 = pytesseract.image_to_string(grey3)
        if score2.endswith('\n'):
            score2[:-2]

        if score1 == ('oo\n'):
            score1 = 0

        if score2 == ('oo\n'):
            score2 = 0

        if score1 == (''):
            score1 = 0

        if score2 == (''):
            score2 = 0

        if score1 == str():
            score1.delete(str())

        if score2 == str():
            score2.delete(str())

        if int(score2) - int(score1) == 10:
            reward = self.GOOD
            print('ok')

        if int(score2) - int(score1) > 10 < 1000:
            reward = self.GOOD
            print('ok')

        if int(score2) - int(score1) == 1000:
            reward = self.GOOD
            print('good')

        if int(score2)  - int(score1) == 2000:
            reward = self.GOOD
            print('great')

        if int(score2)  - int(score1) > 1000 < 2_000:
            reward = self.GOOD
            print('great')

        if int(score2)  - int(score1) == 3000:
            reward = self.GOOD
            print('great')

        if int(score2)  - int(score1) > 3000 < 10_000:
            reward = self.GOOD
            print('fantastic')

        #for the push start = Done.
        screen3 = np.array(ImageGrab.grab(bbox=(790, 690, 1000, 725)))
        grey3 = cv2.cvtColor(screen3, cv2.COLOR_BGR2GRAY)
        prices2 = pytesseract.image_to_string(grey3)

        if prices2.startswith('P'):
            reward = -self.BAD
            time.sleep(0.25)
            if reward == -self.BAD:
                done = True
        else:

            reward = self.GOOD

        return state, new_state, reward, done

    def eyes(self):

      #gets the screen just around the game nothing else.
        self.screen = np.array(ImageGrab.grab(bbox=(620, 328, 1172, 820)))
        self.screen = cv2.resize(self.screen, (80,80))
        cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("image", np.array(screen))
        return self.screen

env = Environment()

ep_rewards = [0]

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class ModifiedTensorBoard(TensorBoard):
    #this is modifying the tensorboard functionality
    #from tensorflow and keras
    #this creates just one log file, whereas normally lots would be created.

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class Deep_stb:

    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # An array with last n steps for training
        self.replay_memory = []

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        #this allows us to use the timesteps function easier in the agent class.
        self.timesteps = timesteps

    def create_model(self):

        if LOAD_MODEL is not None:

            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")

            model = Sequential()

            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            return model

        else:

            model = Sequential()

            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every episode
    def train(self, terminal_state, step):
        
        #This is the UTD or update to data ratio, it'll repeat train
        #20 times which for predict makes 200 predictions.
        for i in range(20):

            # Start training only if certain number of samples is already saved
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return

            #Get rid of the first 64 in replay_mem if exceeds 128
            if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
                self.replay_memory[-64:]

            #take only the last 32 in self.replay_memory
            future_rep_mem = (self.replay_memory[MINIBATCH_SIZE:])

            max_qs = []

            min_qs = 2

            # Get current states from minibatch, then query NN model for Q values
            current_states = np.array([transition[0] for transition in self.replay_memory])/255

            # Get future states from minibatch, then query NN model for Q values
            # When using target network, query it, otherwise main network should be queried
            new_current_states = np.array([transition[3] for transition in self.replay_memory])/255

            #This allows us to calculate future rewards in the last 32 states of replay_memory
            for index, (current_state, action, reward, new_current_state, done) in enumerate(future_rep_mem):

                future_rewards = []
                last_offset = 0
                done = False
                #for each item in range 32 + 1
                for j in range(self.timesteps[-1]+1):
                    #if not done for each time_step that you're on.
                    if not (done + j):
                        if j in self.timesteps: # 1,2,4,8,16,32
                            #It'll keep calculating future_reward_vals till done.
                            if not done:
                                #rewardss over j future_time_steps - current_rewards.
                                future_rewards += list( (([transition[2] for transition in self.replay_memory] + j) - [transition[2] for transition in self.replay_memory]) )
                                last_offset = j
                            else:
                                #If done at this point future_rewards = reward of last time_step - reward.
                                future_rewards += list( (([transition[2] for transition in self.replay_memory] + last_offset) - [transition[2] for transition in self.replay_memory]) )
                    else:
                        done = True

                current_states = np.array([transition[0] for transition in self.replay_memory])/255

                new_current_states = np.array([transition[3] for transition in self.replay_memory])/255


            current_qs_list = self.model.predict(current_states)

            #for each item in range ten, predict on future states, and append those predictions to max_qs.
            for i in range(10):

                future_qs_list = self.target_model.predict(new_current_states)
                max_qs.append(np.array(future_qs_list))

            #for each state in replay_mem rewards = future_rewards.
            for i in self.replay_memory:

                [transition[2] for transition in self.replay_memory] == future_rewards


            X = []
            y = []

            # Now we need to enumerate our batches
            for index, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):

                # If not a terminal state, get new q from future states, otherwise set it to 0
                # almost like with Q Learning, but we use just part of equation here
                if not done:

                    #the max random sample of 2 qs from the 10 qs we calculated.
                    select_f_qs = np.max(random.sample(max_qs, min_qs))

                    #the mean/average value of new_q
                    mean_f_qs = np.mean(select_f_qs)

                    #the variance value of new_q
                    var_f_qs = np.var(select_f_qs)

                    #calculating layer normalization by dividing the square root
                    #of variance of new_q by the new q val plus future reward, subtracted by the mean of new_q.
                    LN_qs = ((select_f_qs) - mean_f_qs)/(np.sqrt(var_f_qs))

                    new_q = reward + DISCOUNT * LN_qs

                else:

                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q
                # And append to our training data
                #literally the images from the game.
                X.append(current_state)
                #the actions we decide to take.               
                y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(np.array(X)/255, np.array(y), batch_size=REPLAY_MEMORY_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard])

            # Update target network counter every episode
            self.target_update_counter += 1

            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

agent = Deep_stb()
env.startGame()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        print('this is the reward:', episode_reward)

        # Every step we update replay memory and update state.
        agent.update_replay_memory((current_state, action, reward, new_state, done))

        current_state = new_state
        step += 1

    #This trains the network after the episode is done.
    agent.train(done, step)

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal to a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
