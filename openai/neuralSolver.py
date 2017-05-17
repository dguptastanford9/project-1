import tensorflow as tf
import cv2
import random
import numpy as np
from collections import deque
import util
from gym import  spaces
from copy import copy
import gym
from gym import wrappers
import gym_ple
from PIL import Image


# -- possible network flavor types ----------------------------------- 
__DEEP_RECURRENT_Q_NETWORK_LSTM = 'deepRecurrentQNetwork'
__DEEP_Q_NETORK = "deepQNetwork"
logs_path = '/tmp/tensorflow_logs/example'
saved_networks_path = '/tmp/saved_networks/'
# -------------------------------------------------------------------

class NeuralSolver():
    
    def __init__(self,
                 modelType="deepQNetwork",
                 gameEnv=None,
                 agentClass=None,
                 epsilon=1,
                 epsilonDecay=.99,
                 gamma=.99,
                 numTraining=1000,
                 batchSize=32,
                 framePerAction=1,
                 numActions=2,
                 replayMemory=50000,
                 game='flappyBird',
                 learningRate=1e-3,
                 learningDecay=1.0,
                 numStepsBeforeSaveModel=10000,
                 numEpisodesRun=10
                 ):
        
        self.gameEnv = gameEnv
        self.modelType = modelType
        self.epsilon = epsilon  # uses random action or predicted action based on epsilon value
        self.epsilonDecay = epsilonDecay  # decay rate for epsilon
        self.gamma = gamma  # reward factor
        self.numTraining = numTraining  # number of examples for training
        self.batchSize = batchSize  # number of data points descent
        self.framePerAction = framePerAction
        self.numActions = numActions
        self.game = game  
        self.explore = self.numTraining  # frames over which to anneal epsilon
        self.replayMemory = replayMemory  # number of previous steps to store
        self.learningRate = learningRate  # learning rate optimization algorithm(RMS/ADAM etc)
        self.learningDecay = learningDecay
        self.numStepsBeforeSaveModel = numStepsBeforeSaveModel  # save model after these number of steps
        self.numEpisodesRun = numEpisodesRun  # total number of episodes we want to execute
        
        self.agent = util.lookup(agentClass, globals())(actionFn=self.getLegalActions)
        
        print("-------- BASIC MODEL HYPER PARAMS USED TO RUN THE MODEL -------------------")
        
        print("STARTING CONV MODEL WITH FOLLOWING PARAMS \n : 1. episilon = ", self.epsilon, \
              " \n 2. epsilonDecay=", self.epsilonDecay, \
              " \n 3. gamma = ", self.gamma, \
              " \n 4. numTraining = ", self.numTraining , \
              " \n 5. batchSize = ", self.batchSize , \
              " \n 6. numActions = ", self.numActions , \
              " \n 7. learningRate = ", self.learningRate , \
              " \n 8. numStepsBeforeSaveModel = ", self.numStepsBeforeSaveModel, \
              " \n 9. numEpisodesRun = ", self.numEpisodesRun 
              )
        
        print("----------------------------------------------------------------------------")
    
    # ------------- OPEN AI RELATED METHODS-------------------------------
    
    def getAction(self, observation):
        return self.agent.getAction(observation)

    def getLegalActions(self, state=None):
        if isinstance(self.gameEnv.action_space, spaces.discrete.Discrete):
            return tuple(range(self.gameEnv.action_space.n))
        raise Exception, 'Action type not supported: %s' % type(self.gameEnv.action_space)

    def observeTransition(self, old_obs, action, observation, reward):
        self.agent.observeTransition(old_obs, action, observation, reward)
        
    def prepareState(self, observation):
        
        if isinstance(self.gameEnv.observation_space, spaces.discrete.Discrete):
            pass
        
        elif isinstance(self.gameEnv.observation_space, spaces.box.Box):
#             if self.discretize:
#                 observation = self.toBins(observation)
            observation = tuple(observation)
        
        else:
            raise Exception, 'Observation type not supported: %s' % type(self.gameEnv.observation_space)
    
#         if self.sim_env:
#             observation = (copy(self.gameEnv), observation)
        return observation
    
    # ------------- END OF OPEN AI RELATED METHODS-------------------------------
    
    
    def solve_network(self):
        print('start solving network')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    
    ## ----------------Create Model ------------------------------
    
    def createNetwork(self):
        
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.numActions])
        b_fc2 = self.bias_variable([self.numActions])

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # y_out layer
        y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, y_out, h_fc1
    
    ## --- Finished creating model ----------------------------------

    def trainNetwork(self, s, model_yout, h_fc1, sess, displayGraphics=True):
        
        # # ----- Basic tensor flow set up -------------------------## 
        
        a = tf.placeholder("float", [None, self.numActions])
        y = tf.placeholder("float", [None])
        predictedCost = tf.reduce_sum(tf.multiply(model_yout, a), axis=1)
        cost = tf.reduce_mean(tf.square(y - predictedCost))  # this is more of regression kind of cost function 
        tf.summary.scalar("loss", cost)  # Create a summary to monitor cost tensor
        merged_summary_op = tf.summary.merge_all()  # Merge all summaries into a single op
        optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)  # TODO make this configurable
        
        # # --------- End of Basic Tensor flow set up ----------------------- ##
        
        # this will store all the event for replay memory
        D = deque()
        
        # saving and loading networks .. this will by default save all tensor flow Variables (placehoders)
        saver = tf.train.Saver()
        # initialize session for  tensor flow
        sess.run(tf.global_variables_initializer())
        # op to write logs to Tensor board
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        #---- set up to save and load  networks
        checkpoint = tf.train.get_checkpoint_state(saved_networks_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # -----start training ---------
        t = 0  # number of transitions 
        
        for episodeNum in range(self.numEpisodesRun + 100):  # TODO : number of episodes can be tuned 
            
            # ---- open ai game emulator integration  with initial bootstrapping------
          
            initialColoredObservation = self.gameEnv.reset()
            # initialColoredObservation = self.prepareState(initialColoredObservation)
        
            self.agent.startEpisode() 
            self.gameEnv.render(close=not displayGraphics)
            gameAction = random.choice(self.getLegalActions(initialColoredObservation))  # choose a scalar randomly from a set of legal actions
            initialColoredObservation, _, _, _ = self.gameEnv.step(gameAction)  # pass in scalar action to get output
            
            # do some pre-processing on  image
            imageBackgroundGray = self.convertImageBackgroubtToGray(initialColoredObservation)
            currentLastFourImageFrames = np.stack((imageBackgroundGray, imageBackgroundGray, imageBackgroundGray, imageBackgroundGray), axis=2)
            done = False
            
            while not done :
                
                # this will evaluate tensor flow model for this current state !!
                # similar to sess.run(y_out,feed_dict={X:x,is_training:True})
                yout_t = model_yout.eval(feed_dict={s: [currentLastFourImageFrames]})[0]
                actionVector = np.zeros([self.numActions])
                action_index = 0
                
                # epsilon value will be initialized with a higher value when we have a decent enough model to train
                 
                if random.random() <= self.epsilon:
                    action_index = random.randrange(self.numActions)
                    actionVector[random.randrange(self.numActions)] = 1
                else:
                    action_index = np.argmax(yout_t)
                    actionVector[action_index] = 1
                
                    
                # run the selected action and observe next state and reward
                
                # currentColoredImageObservation = self.prepareState(currentColoredImageObservation)
                nextColoredImageObservation, reward, isEpisodeDone, _ = self.gameEnv.step(action_index)  # Fix this with open ai
                nextImageBackgroundGray = cv2.cvtColor(cv2.resize(nextColoredImageObservation, (80, 80)), cv2.COLOR_BGR2GRAY)
                _, nextImageBackgroundGray = cv2.threshold(nextImageBackgroundGray, 1, 255, cv2.THRESH_BINARY)
                nextImageBackgroundGray = np.reshape(nextImageBackgroundGray, (80, 80, 1))
    
                nextLastFourImageFrames = np.append(nextImageBackgroundGray, currentLastFourImageFrames[:, :, 0:3], axis=2)
    
                # store the transition in D
                
                D.append((currentLastFourImageFrames, actionVector, reward, nextLastFourImageFrames, isEpisodeDone))
                if len(D) > self.replayMemory:
                    D.popleft()
                
                if isEpisodeDone:
                    done = True
                        
                # only train if done observing
                if t > self.numTraining:
                    if (t == self.numTraining + 1) :
                        print("****** Beginning training the model as we have reached the threshold for explore *******")
                        
                    # sample a batch to train on
                    minibatch = random.sample(D, self.batchSize)
                    # get the batch variables
                    current_state_image_minibatch = [d[0] for d in minibatch]
                    current_action_minibatch = [d[1] for d in minibatch]
                    current_rewards_minibatch = [d[2] for d in minibatch]
                    next_state_image_minibatch = [d[3] for d in minibatch]
                    
                    
                    y_batch = []
                    readout_j1_batch = model_yout.eval(feed_dict={s: next_state_image_minibatch})
                    for i in range(0, len(minibatch)):
                        isEpisodeDone = minibatch[i][4]
                        # if isEpisodeDone, only equals reward , because thats the max we can achieve anyways now  
                        if isEpisodeDone:
                            y_batch.append(current_rewards_minibatch[i])
                        else:
                            y_batch.append(current_rewards_minibatch[i] + self.gamma * np.max(readout_j1_batch[i]))
    
                    # perform gradient step .. tensorflow magic !!
                    
                    # Run optimization op (backprop), cost op (to get loss value)
                    # and summary nodes
                    _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={
                        y: y_batch,
                        a: current_action_minibatch,
                        s: current_state_image_minibatch})
                    summary_writer.add_summary(summary, t - self.numTraining + 1)
                    

                # update the old values
                currentLastFourImageFrames = nextLastFourImageFrames
                # currentColoredImageObservation = nextColoredImageObservation
                
                # update t to track how many training or observations sampled
                t += 1
                
                # save progress every 10000 iterations
                if t % self.numStepsBeforeSaveModel == 0:
                    saver.save(sess, saved_networks_path + self.game + '-dqn', global_step=t)
    
                # print info
                state = ""
                if t <= self.numTraining:
                    state = "observe"
                elif t > self.numTraining and t <= self.numTraining + self.explore:
                    state = "explore"
                else:
                    state = "train"
    
                print("TIMESTEP", t, "/ EPISODE", episodeNum, "/ STATE", state, \
                      "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                      "/ Q_MAX %e" % np.max(yout_t))

                if reward > 0:
                    print("GREAT SUCCESS! reward = ", reward) 

            # scale down epsilon as we train
            #this is a linear decay self.epsilon -= self.epsilonDecay  / self.numEpisodesRun
            self.epsilon *= self.epsilonDecay
            if episodeNum > self.numEpisodesRun:
                self.epsilon = 0
        self.gameEnv.close()
  
    
    def playGame(self):
        sess = tf.InteractiveSession()
        s, y_out, h_fc1 = self.createNetwork()
        self.trainNetwork(s, y_out, h_fc1, sess)
        
    
    def convertImageBackgroubtToGray(self, currentImageOutputColored):
        imageWithBackgroundGray = cv2.cvtColor(cv2.resize(currentImageOutputColored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, imageWithBackgroundGray = cv2.threshold(imageWithBackgroundGray, 1, 255, cv2.THRESH_BINARY)
        return imageWithBackgroundGray    
    
    
        
    def main(self):
        self.playGame()
