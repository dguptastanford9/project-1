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
#logs_path = '/tmp/'
#saved_networks_path = '/tmp/'
# -------------------------------------------------------------------

class DDQN():
    
    def __init__(self,
                 modelType="DDQN",
                 gameEnv=None,
                 agentClass=None,
                 epsilon=1,
                 epsilonDecay=.99,
                 gamma=.99,
                 numTraining=1000,
                 batchSize=32,
                 framePerAction=1,
                 numActions=2,
                 replayMemory=40000,
                 game='flappyBird',
                 learningRate=1e-3,
                 learningDecay=1.0,
                 numStepsBeforeSaveModel=10000,
                 numEpisodesRun=10,
                 mode='cpu',
                 displayGraphics=False,
                 numEpsilonCycle=7,
                 updateTargetNetworkAfterIterations=10000,
                 useRewardClippingMode=False
                 ):
        
        self.gameEnv = gameEnv
        self.modelType = modelType
        self.epsilon = epsilon  # uses random action or predicted action based on epsilon value
        self.epsilon0 = epsilon
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
        self.numStepsBeforeSaveModel = numStepsBeforeSaveModel  # save model after these number of steps
        self.numEpisodesRun = numEpisodesRun  # total number of episodes we want to execute
        self.mode = mode
        # self.agent = util.lookup(agentClass, globals())(actionFn=self.getLegalActions)
        self.displayGraphics = displayGraphics
        self.numEpsilonCycle = .5 + numEpsilonCycle  # number of time epsilon will decay to 0 throughout the training cycle
        self.updateTargetNetworkAfterIterations = updateTargetNetworkAfterIterations
        self.debug = False  # will log some extra print statements to standard out
        self.useRewardClippingMode = useRewardClippingMode
        
        print("-------- BASIC MODEL HYPER PARAMS USED TO RUN THE MODEL -------------------")
        
        print("STARTING DDQN MODEL WITH FOLLOWING PARAMS: \n 1. episilon = ", self.epsilon, \
              " \n 2. epsilonDecay=", self.epsilonDecay, \
              " \n 3. gamma = ", self.gamma, \
              " \n 4. numTraining = ", self.numTraining , \
              " \n 5. batchSize = ", self.batchSize , \
              " \n 6. numActions = ", self.numActions , \
              " \n 7. learningRate = ", self.learningRate , \
              " \n 8. numStepsBeforeSaveModel = ", self.numStepsBeforeSaveModel, \
              " \n 9. numEpisodesRun = ", self.numEpisodesRun, \
              " \n 10. numEpsilonCycle = ", self.numEpsilonCycle , \
              " \n 11. updateTargetNetworkAfterIterations = ", self.updateTargetNetworkAfterIterations,
              " \n 12 . useRewardClippingMode = ", self.useRewardClippingMode
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

    
    def updateTargetNetwork(self):
        for targetVariableName in self.targetToSourceVariableNames:
            sourceTensorVariableName = self.targetToSourceVariableNames[targetVariableName]
            sourceWeight = self.sess.run(self.sourceNameToVariableDict[sourceTensorVariableName])
            if len(sourceWeight.shape) == 4 and self.debug :
                print('Source variable :{} value being applied to target {} : \n'.
                      format(sourceTensorVariableName, sourceWeight[0, 0, 0, :]))
            self.sess.run(self.t_w_assign_op[targetVariableName], feed_dict={self.t_w_input[targetVariableName] : sourceWeight})    
            
            if len(sourceWeight.shape) == 4 and self.debug :
                print('Target Variable : {} after update from source weights {} :\n'.
                      format(targetVariableName, self.sess.run(self.targetNameToVariableDict[targetVariableName])[0, 0, 0, :]))
                
    def createNetwork(self):

        with tf.device('/cpu:0' if self.mode == 'cpu' else '/gpu:0'):
           
            with tf.variable_scope('prediction') :
                # input layer
                self.inputImageVector = tf.placeholder("float", [None, 80, 80, 4])
                # hidden layers
                h_conv1 = tf.layers.conv2d(self.inputImageVector, 32, 8, 4, activation=tf.nn.relu, use_bias=True, name="CONV1")
                h_conv2 = tf.layers.conv2d(h_conv1, 64, 4, 2, activation=tf.nn.relu, use_bias=True, name="CONV2")
                h_conv3 = tf.layers.conv2d(h_conv2, 64, 3, 1, activation=tf.nn.relu, use_bias=True, name="CONV3")
                flatten = tf.reshape(h_conv3, [-1, 2304])
                # fully connected layers
                h_fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, use_bias=True, name="FC1")
                self.fc_out = tf.layers.dense(h_fc1, 2, activation=None, use_bias=True, name="FC_OUT")
                self.fc_out_action = tf.argmax(self.fc_out, dimension=1)
               
            with tf.variable_scope('target'):
                
                self.inputImageVector_target = tf.placeholder("float", [None, 80, 80, 4])
                h_conv1_target = tf.layers.conv2d(self.inputImageVector_target, 32, 8, 4, activation=tf.nn.relu, use_bias=True, name="CONV1_TARGET")
                h_conv2_target = tf.layers.conv2d(h_conv1_target, 64, 4, 2, activation=tf.nn.relu, use_bias=True, name="CONV2_TARGET")
                h_conv3_target = tf.layers.conv2d(h_conv2_target, 64, 3, 1, activation=tf.nn.relu, use_bias=True, name="CONV3_TARGET")
                flatten_target = tf.reshape(h_conv3_target, [-1, 2304])
                # fully connected layers
                h_fc1_target = tf.layers.dense(flatten_target, 512, activation=tf.nn.relu, use_bias=True, name="FC1_TARGET")
                self.fc_out_target = tf.layers.dense(h_fc1_target, 2, activation=None, use_bias=True, name="FC_OUT_TARGET")
                
                self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')   
                self.target_q_with_idx = tf.gather_nd(self.fc_out_target, self.target_q_idx)   
                
                self.sourceToTargetVariableNames = { 'prediction/CONV1/kernel:0' : 'target/CONV1_TARGET/kernel:0',
                               'prediction/CONV2/kernel:0' : 'target/CONV2_TARGET/kernel:0',
                               'prediction/CONV3/kernel:0' : 'target/CONV3_TARGET/kernel:0',
                               'prediction/FC1/kernel:0'   : 'target/FC1_TARGET/kernel:0' ,
                               'prediction/FC_OUT/kernel:0': 'target/FC_OUT_TARGET/kernel:0'
                            }
                
                self.targetToSourceVariableNames = {}
                for sourceVariableName, targetVariableName in self.sourceToTargetVariableNames.items():
                    self.targetToSourceVariableNames[targetVariableName] = sourceVariableName
                
                self.sourceNameToVariableDict = {}
                for sVariable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prediction'):
                    if sVariable.name in self.sourceToTargetVariableNames:
                        self.sourceNameToVariableDict[sVariable.name] = sVariable
                
                self.targetNameToVariableDict = {}
                print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target'))
                for tVariable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target'):
                    if tVariable.name in self.targetToSourceVariableNames:
                        self.targetNameToVariableDict[tVariable.name] = tVariable
                
            with tf.variable_scope('optimizer'):    
                
                self.predictedActionScoreVector = tf.placeholder("float", [None, self.numActions])  # min_batch_size * numActions
                self.predictedScore = tf.reduce_sum(tf.multiply(self.fc_out, self.predictedActionScoreVector), axis=1)  # scalar
                self.targetQ = tf.placeholder("float", [None])  # scalar value  --> this will be calculated by the
                
                if  self.useRewardClippingMode:
                    self.loss = tf.reduce_mean(self.clipped_error(self.targetQ - self.predictedScore)) 
                else :
                    self.loss = tf.reduce_mean(tf.square(self.targetQ - self.predictedScore))
                
                tf.summary.scalar("loss", self.loss)  # Create predictedActionScoreVector summary to monitor cost tensor 
                tf.summary.scalar("Predicted_Q-Value", tf.reduce_mean(self.predictedScore))
                tf.summary.scalar("Actual_Q-Value", tf.reduce_mean(self.targetQ))
                self.merged_summary_op = tf.summary.merge_all()  # Merge all summaries into predictedActionScoreVector single operation
                self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)  # TODO make this configurable
            
            with tf.variable_scope('pred_to_target') :
                
                self.t_w_input = {}
                self.t_w_assign_op = {}
                namePrefix = "variable"
                num = 1
                for targetVariableName in self.targetToSourceVariableNames:
                    self.t_w_input[targetVariableName] = tf.placeholder('float32', self.targetNameToVariableDict[targetVariableName].get_shape().as_list(), name=namePrefix + str(num))
                    self.t_w_assign_op[targetVariableName] = self.targetNameToVariableDict[targetVariableName].assign(self.t_w_input[targetVariableName])
                    num += 1
                
                        
    
    ## --- Finished creating model ----------------------------------

    def trainNetwork(self):
        
        # ----------------- tensor flow related setup -------------------
        
        saver = tf.train.Saver()  # saving and loading networks .. this will by default save all tensor flow Variables (placeholders)
        self.sess.run(tf.global_variables_initializer())  # initialize session for  tensor flow
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # op to write logs to Tensor board
        
        checkpoint = tf.train.get_checkpoint_state(saved_networks_path)  # set up to save and load  networks
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.updateTargetNetwork()
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            
        # --------------------------------start training ----------------
        
        replayMemoryQueue = deque()  # this will store all the event for replay memory
        numIterations = 0  # number of iterations 
        avgReward = 0
        updateTargetThresh = 10000
        for episodeNum in range(self.numEpisodesRun + 100):  # TODO : number of episodes can be tuned 
            
            # ---- open-ai game emulator integration  with initial bootstrapping------
            
            initialColoredObservation = self.gameEnv.reset()
            self.gameEnv.render(close=not self.displayGraphics)
            gameAction = random.choice(self.getLegalActions(initialColoredObservation))  # choose predictedActionScoreVector scalar randomly from predictedActionScoreVector set of legal actions
            
            initialColoredObservation, _, _, _ = self.gameEnv.step(gameAction)  # pass in scalar action to get output
            imageBackgroundGray = self.convertImageBackgroundToGray(initialColoredObservation)  # do pre-processing on image
            currentLastFourImageFrames = np.stack((imageBackgroundGray, imageBackgroundGray, imageBackgroundGray, imageBackgroundGray), axis=2)
            
            episode_pos_reward = 0  # count the number of positive rewards received
            done = False
            modelSaved = False
            
            while not done :
                            
                actionVector = np.zeros([self.numActions])
                action_index = 0
                
                if random.random() <= self.epsilon:  # we will gradually decrease epsilon as we explore 
                    action_index = random.randrange(self.numActions)
                    actionVector[action_index] = 1
                else:
                    yout_t = self.sess.run(self.fc_out, feed_dict={self.inputImageVector: [currentLastFourImageFrames]})  # <-- get action from current state prediction
                    action_index = np.argmax(yout_t)
                    actionVector[action_index] = 1
                
                nextColoredImageObservation, reward, isEpisodeDone, _ = self.gameEnv.step(action_index)  # <--run the selected action and observe next state and reward
               
                if reward != 0:  # <-- process reward to be [-1, 0, 1]
                    reward /= math.fabs(reward)

                nextImageBackgroundGray = self.convertImageBackgroundToGray(nextColoredImageObservation)  # <-- do pre-processing on image
                nextImageBackgroundGray = np.reshape(nextImageBackgroundGray, (80, 80, 1))
                nextLastFourImageFrames = np.append(nextImageBackgroundGray, currentLastFourImageFrames[:, :, 0:3], axis=2)  # <-- stack last 4 image frames for next state 
                
                replayMemoryQueue.append((currentLastFourImageFrames, actionVector, reward, nextLastFourImageFrames, 1 if isEpisodeDone else 0))  # <-- store the transition in replayMemoryQueue
               
                if len(replayMemoryQueue) > self.replayMemory:
                    replayMemoryQueue.popleft()
                
                done = isEpisodeDone
                 
                if numIterations > self.numTraining:  # <-- only train/optimize if done observing [ if iterations > numTraining threshold -x option]
                        
                    minibatch = random.sample(replayMemoryQueue, self.batchSize)  # min-batch to perform optimization
                    current_state_image_minibatch = [d[0] for d in minibatch]  # get batch of stacked image frames 
                    current_action_minibatch = [d[1] for d in minibatch]  # min batch for actions performed on each stacked image frame 
                    current_rewards_minibatch = [d[2] for d in minibatch]  # min batch for rewards 
                    next_state_image_minibatch = [d[3] for d in minibatch]  # get next batch of stacked image frames 
                    
                    targetQ = []
                    if self.useRewardClippingMode :
                        current_rewards_minibatch = np.clip(current_rewards_minibatch, -1, 1)
                        
                    next_state_reward_eval = self.sess.run(self.fc_out_action, feed_dict={self.inputImageVector: next_state_image_minibatch})  # <-- get the next state reward 
                    next_state__reward_dqn_eval = self.sess.run(self.target_q_with_idx, feed_dict={  # <----------------------------- Double Q Learning , we don't do max  
                                                           self.inputImageVector_target: next_state_image_minibatch,
                                                           self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(next_state_reward_eval)]
                    })
                    batchEpisodeDone = [d[4] for d in minibatch]
                    targetQ = (1. - np.array(batchEpisodeDone)) * self.gamma * next_state__reward_dqn_eval + current_rewards_minibatch
                    _, _, _, summary = self.sess.run([self.optimizer, self.fc_out, self.loss, self.merged_summary_op], feed_dict={  # <-- Do the optimization step
                                              self.targetQ: targetQ,  # and summary nodes    
                                              self.predictedActionScoreVector: current_action_minibatch,
                                              self.inputImageVector: current_state_image_minibatch    
                                           })
                    summary_writer.add_summary(summary, numIterations - self.numTraining + 1)  #  <-- tensor board stuff 
                    
                    # hyperparameter based on paper https://deepmind.com/research/dqn/
                    if numIterations == updateTargetThresh: 
                        print('------Updating target network ---------------') 
                        self.updateTargetNetwork()   
                        updateTargetThresh += int((avgReward+1) * 1500)

                # update the old values
                currentLastFourImageFrames = nextLastFourImageFrames
                # currentColoredImageObservation = nextColoredImageObservation
                
                # update numIterations to track how many training or observations sampled
                numIterations += 1
                
                # save progress every epsilon cycle 
                if not modelSaved  and (episodeNum + int(self.numEpisodesRun / self.numEpsilonCycle / 2)) % int(self.numEpisodesRun / self.numEpsilonCycle + 1) == 0: 
                    print("****** Saving Model at episode ", episodeNum, " iteration ", numIterations, " epsilon ", self.epsilon, " ******")
                    saver.save(self.sess, saved_networks_path + self.game + '-ddqn', global_step=numIterations)
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
               
            
            # --------------- End of Episode -----------------------------# 
            print("EPISODE", episodeNum, \
                    "/ POSITIVE REWARDS", episode_pos_reward, \
                    "/ STATE", state, \
                    "/ EPSILON", self.epsilon, \
                    "/ Iteration number", numIterations , \
                    "/ RUNNING AVERAGE REWARDS", avgReward)
            avgReward = .99 * avgReward + .01 * episode_pos_reward 
            # scale down epsilon as we train ( epsilon oscillating decay formula)
            if state != "observe":
                self.epsilon = self.epsilon0 * np.power(self.epsilonDecay, episodeNum) * \
                (1 + np.cos(2 * np.pi * episodeNum / (self.numEpisodesRun / self.numEpsilonCycle))) / 2 + .001

            if episodeNum > self.numEpisodesRun:
                self.epsilon = 0
        # ---- all done , close game environment ----
        self.gameEnv.close()
  
    
    def playGame(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.createNetwork()
        self.trainNetwork()

    def convertImageBackgroundToGray(self, currentImageOutputColored):
        imgGray = cv2.cvtColor(currentImageOutputColored, cv2.COLOR_BGR2GRAY)
        block_size = 3
        imgGaussGray = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    cv2.THRESH_BINARY, block_size, 2)
        imgResize = cv2.resize(imgGaussGray, (80, 80))
        imgNormalized = np.divide(imgResize, 255)
        return imgNormalized
    
    def updateTargetNetworkSlowlyButFrequently(self):
        print('--- This updates the target network to source network ---')
    
    
    
    
    def clipped_error(self,x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
       
       
    def main(self):
        self.playGame()
