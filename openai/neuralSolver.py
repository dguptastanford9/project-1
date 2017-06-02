import tensorflow as tf
import cv2
import random, math
import numpy as np
from collections import deque
import util
from gym import  spaces
from copy import copy
import gym
from gym import wrappers
import gym_ple
from PIL import Image
import png

# -- possible network flavor types ----------------------------------- 
__DEEP_RECURRENT_Q_NETWORK_LSTM = 'deepRecurrentQNetwork'
__DEEP_Q_NETORK = "deepQNetwork"
logs_path = '/home/tf/tensorflow_logs/example'
saved_networks_path = '/home/tf/saved_networks/'
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
                 numStepsBeforeSaveModel=100000,
                 replayMemory=40000,
                 game='flappyBird',
                 learningRate=1e-3,
                 learningDecay=1.0,
                 numEpsilonCycle=7,
                 numEpisodesRun=10,
                 mode='cpu',
		         displayGraphics=False
                 ):
        
        self.gameEnv = gameEnv
        self.modelType = modelType
        self.epsilon = epsilon  # uses random action or predicted action based on epsilon value
        self.epsilon0 = epsilon # starting epsilon value 
        self.epsilonDecay = epsilonDecay  # decay rate for epsilon
        self.gamma = gamma  # reward factor
        self.numTraining = numTraining  #  number of examples for training
        self.batchSize = batchSize  # number of data points descent
        self.framePerAction = framePerAction
        self.numActions = numActions
        self.game = game  
        self.explore = self.numTraining  # frames over which to anneal epsilon
        self.replayMemory = replayMemory  # number of previous steps to store
        self.learningRate = learningRate  # learning rate optimization algorithm(RMS/ADAM etc)
        self.learningDecay = learningDecay
        self.numEpsilonCycle = .5 + numEpsilonCycle # number of time epsilon will decay to 0 throughout the training cycle
        self.numEpisodesRun = numEpisodesRun  # total number of episodes we want to execute
        self.mode=mode
        self.numStepsBeforeSaveModel = numStepsBeforeSaveModel
        #self.agent = util.lookup(agentClass, globals())(actionFn=self.getLegalActions)
        self.displayGraphics = displayGraphics

        print("-------- BASIC MODEL HYPER PARAMS USED TO RUN THE MODEL -------------------")
        
        print("STARTING CONV MODEL WITH FOLLOWING PARAMS: \n 1. episilon = ", self.epsilon, \
              " \n 2. epsilonDecay=", self.epsilonDecay, \
              " \n 3. gamma = ", self.gamma, \
              " \n 4. numTraining = ", self.numTraining , \
              " \n 5. batchSize = ", self.batchSize , \
              " \n 6. numActions = ", self.numActions , \
              " \n 7. learningRate = ", self.learningRate , \
              " \n 8. numStepsBeforeSaveModel = ", self.numStepsBeforeSaveModel, \
              " \n 9. numEpisodesRun = ", self.numEpisodesRun, \
              " \n 10. numEpsilonCycle = ", numEpsilonCycle
              )
        
        print("----------------------------------------------------------------------------")
    
    # ------------- OPEN AI RELATED METHODS-------------------------------
    
    def getAction(self, observation):
        return self.agent.getAction(observation)

    def getLegalActions(self, state=None):
        if isinstance(self.gameEnv.action_space, spaces.discrete.Discrete):
            return tuple(range(self.gameEnv.action_space.n))
        raise Exception('Action type not supported: %s' % type(self.gameEnv.action_space))

    def observeTransition(self, old_obs, action, observation, reward):
        self.agent.observeTransition(old_obs, action, observation, reward)
        
    def prepareState(self, observation):
        
        if isinstance(self.gameEnv.observation_space, spaces.discrete.Discrete):
            pass
        
        elif isinstance(self.gameEnv.observation_space, spaces.box.Box):
            observation = tuple(observation)
        
        else:
            raise Exception('Observation type not supported: %s' % type(self.gameEnv.observation_space))
    
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

        inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op = [None] * 7
        
        with tf.device('/cpu:0' if self.mode == 'cpu' else '/gpu:0'):
           
             

            # input layer
            inputImageVector = tf.placeholder("float", [None, 80, 80, 4])
    
            # hidden layers
            h_conv1 = tf.layers.conv2d(inputImageVector, 32, 8, 4, activation=tf.nn.relu, use_bias=True, name="CONV1")

            h_conv2 = tf.layers.conv2d(h_conv1, 64, 4, 2, activation=tf.nn.relu, use_bias=True, name="CONV2")
            
            h_conv3 = tf.layers.conv2d(h_conv2, 64, 3, 1, activation=tf.nn.relu, use_bias=True, name="CONV3")
            
            print(h_conv3.shape)
            flatten = tf.reshape(h_conv3, [-1, 2304])

            # fully connected layers
            h_fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, use_bias=True, name="FC1")
            
            fc_out = tf.layers.dense(h_fc1, 2, activation=None, use_bias=True, name="FC_OUT")

            
            
            predictedActionScoreVector = tf.placeholder("float", [None, self.numActions])
        
            actualScore = tf.placeholder("float", [None])  # scalar value
            
            predictedScore = tf.reduce_sum(tf.multiply(fc_out, predictedActionScoreVector), axis=1)  # scalar
            
            cost = tf.reduce_mean(tf.square(actualScore - predictedScore))  # this is more of regression kind of cost function 
            
            tf.summary.scalar("loss", cost)  # Create predictedActionScoreVector summary to monitor cost tensor
            tf.summary.scalar("Predicted_Q-Value", tf.reduce_mean(predictedScore))
            tf.summary.scalar("Actual_Q-Value", tf.reduce_mean(actualScore))
            merged_summary_op = tf.summary.merge_all()  # Merge all summaries into predictedActionScoreVector single operation
            
            optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)  # TODO make this configurable
            
        return inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op   
    
    ## --- Finished creating model ----------------------------------

    def trainNetwork(self,
                    sess,
                    inputImageVector,
                    fc_out,
                    predictedActionScoreVector,
                    actualScore,
                    cost,
                    optimizer,
                    merged_summary_op):
        
        # ----------------- tensor flow related setup -------------------
        
        saver = tf.train.Saver()  # saving and loading networks .. this will by default save all tensor flow Variables (placeholders)
        sess.run(tf.global_variables_initializer())  # initialize session for  tensor flow
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # op to write logs to Tensor board
        
        checkpoint = tf.train.get_checkpoint_state(saved_networks_path)  # set up to save and load  networks
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            
        # --------------------------------start training ----------------
        
        replayMemoryQueue = deque()  # this will store all the event for replay memory
        numIterations = 0  # number of iterations 
        
        for episodeNum in range(self.numEpisodesRun + 100):  # TODO : number of episodes can be tuned 
            
            # ---- open-ai game emulator integration  with initial bootstrapping------
            
            initialColoredObservation = self.gameEnv.reset()
            self.gameEnv.render(close=not self.displayGraphics)
            gameAction = random.choice(self.getLegalActions(initialColoredObservation))  # choose predictedActionScoreVector scalar randomly from predictedActionScoreVector set of legal actions
            
            initialColoredObservation, _, _, _ = self.gameEnv.step(gameAction)  # pass in scalar action to get output
            imageBackgroundGray = self.convertImageBackgroundToGray(initialColoredObservation)  # do pre-processing on image
            currentLastFourImageFrames = np.stack((imageBackgroundGray, imageBackgroundGray, imageBackgroundGray, imageBackgroundGray), axis=2)
            
            episode_pos_reward = 0 #count the number of positive rewards received
            done = False
            modelSaved = False
            
            while not done :
                            
                #yout_t = sess.run(fc_out,feed_dict={inputImageVector: [currentLastFourImageFrames]})  # similar to sess.run(y_out,feed_dict={X:x,is_training:True})
                actionVector = np.zeros([self.numActions])
                action_index = 0
                 
                if random.random() <= self.epsilon:  # we will gradually decrease epsilon as we explore 
                    action_index = random.randrange(self.numActions)
                    actionVector[random.randrange(self.numActions)] = 1
                else:
                    yout_t = sess.run(fc_out,feed_dict={inputImageVector: [currentLastFourImageFrames]})  # similar to sess.run(y_out,feed_dict={X:x,is_training:True})
                    action_index = np.argmax(yout_t)
                    actionVector[action_index] = 1
                
                nextColoredImageObservation, reward, isEpisodeDone, _ = self.gameEnv.step(action_index)  # run the selected action and observe next state and reward
               
                #process reward to be [-1, 0, 1]
                if reward != 0:
                    reward /= math.fabs(reward)

                nextImageBackgroundGray = self.convertImageBackgroundToGray(nextColoredImageObservation)  # do pre-processing on image
                nextImageBackgroundGray = np.reshape(nextImageBackgroundGray, (80,80,1) )
                nextLastFourImageFrames = np.append(nextImageBackgroundGray, currentLastFourImageFrames[:, :, 0:3], axis=2)  # stack last 4 image frames 
                #nextLastFourImageFrames = np.append(currentLastFourImageFrames[:, :, 0:3], nextImageBackgroundGray, axis=2)  # stack last 4 image frames 
                
                replayMemoryQueue.append((currentLastFourImageFrames, actionVector, reward, nextLastFourImageFrames, isEpisodeDone))  # store the transition in replayMemoryQueue
                if len(replayMemoryQueue) > self.replayMemory:
                    replayMemoryQueue.popleft()
                
                if isEpisodeDone:
                    done = True
                        
                if numIterations > self.numTraining:  # only train if done observing
                    
                    if (numIterations == self.numTraining + 1) :
                        print("****** Beginning training the model as we have reached the threshold for explore *******")
                        
                    minibatch = random.sample(replayMemoryQueue, self.batchSize)  # min-batch to perform optimization
                    current_state_image_minibatch = [d[0] for d in minibatch]  # get batch of stacked image frames 
                    current_action_minibatch = [d[1] for d in minibatch]  # min batch for actions performed on each stacked image frame 
                    current_rewards_minibatch = [d[2] for d in minibatch]  # min batch for rewards 
                    next_state_image_minibatch = [d[3] for d in minibatch]  # get next batch of stacked image frames 
                    
                    qfunction = []
                    next_state_reward_eval = sess.run(fc_out,feed_dict={inputImageVector: next_state_image_minibatch})
                    
                    for i in range(0, len(minibatch)):
                        isEpisodeDone = minibatch[i][4]
                        # if isEpisodeDone, only equals reward , because thats the max we can achieve anyways now  
                        if isEpisodeDone:
                            qfunction.append(current_rewards_minibatch[i])
                        else:
                            qfunction.append(current_rewards_minibatch[i] + self.gamma * np.max(next_state_reward_eval[i]))
    
                    # perform gradient step .. tensorflow magic !!
                    # Run optimization op (backprop), cost op (to get loss value)
                    # and summary nodes
                    _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={
                        actualScore: qfunction,
                        predictedActionScoreVector: current_action_minibatch,
                        inputImageVector: current_state_image_minibatch})
                    summary_writer.add_summary(summary, numIterations - self.numTraining + 1)
                    

                # update the old values
                currentLastFourImageFrames = nextLastFourImageFrames
                # currentColoredImageObservation = nextColoredImageObservation
                
                # update numIterations to track how many training or observations sampled
                numIterations += 1
                
                # save progress every epsilon cycle 
                if not modelSaved and (episodeNum + int(self.numEpisodesRun / self.numEpsilonCycle / 2)) % int(self.numEpisodesRun / self.numEpsilonCycle + 1) == 0: 
                    print("****** Saving Model at episode ", episodeNum, " iteration ", numIterations, " epsilon ", self.epsilon, " ******")
                    saver.save(sess, saved_networks_path + self.game + '-dqn', global_step=numIterations)
                    modelSaved = True
    
                # print info
                state = ""
                if numIterations <= self.numTraining:
                    state = "observe"
                elif numIterations > self.numTraining and numIterations <= self.numTraining + self.explore:
                    state = "explore"
                else:
                    state = "train"
    
                if reward > 0:
                    episode_pos_reward += 1
                '''
                print("TIMESTEP", numIterations, "/ EPISODE", episodeNum, "/ STATE", state, \
                      "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                      "/ Q_MAX %e" % np.max(yout_t))
                '''
            #if episode_pos_reward > 0:
            print("EPISODE", episodeNum, \
                "/ POSITIVE REWARDS", episode_pos_reward, \
                "/ STATE", state, \
                "/ EPSILON", self.epsilon,\
                "/ Iteration number",numIterations)
            
            # scale down epsilon as we train
            # this is predictedActionScoreVector linear decay self.epsilon -= self.epsilonDecay  / self.numEpisodesRun
            if state != "observe":
                #self.epsilon *= self.epsilonDecay
                self.epsilon = self.epsilon0 * np.power(self.epsilonDecay, episodeNum) * (1 + np.cos(2 * np.pi * episodeNum / (self.numEpisodesRun / self.numEpsilonCycle))) / 2 + .001

            if episodeNum > self.numEpisodesRun:
                self.epsilon = 0
        self.gameEnv.close()
  
    
    def playGame(self):
        #sess = tf.InteractiveSession()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op = self.createNetwork()
        self.trainNetwork(sess, inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op)
        
    
    def convertImageBackgroundToGray(self, currentImageOutputColored):
        #png.from_array(currentImageOutputColored, "RGB").save("flappy0.png")
        imgGray = cv2.cvtColor(currentImageOutputColored, cv2.COLOR_BGR2GRAY)
        #png.from_array(imgGray, "L").save("flappy2.png")
        block_size = 3
        #block_size = 93
        #for block_size in range(3,99,2):
        imgGaussGray = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,block_size,2)
        #png.from_array(imgGaussGray, "L").save("flappy"+str(block_size)+".png")
        imgResize = cv2.resize(imgGaussGray, (80, 80))
        imgNormalized = np.divide(imgResize, 255)
        #print(imgResize)
        #png.from_array(imgResize, "L").save("flappy1.png")
        return imgNormalized
    
    
        
    def main(self):
        self.playGame()
