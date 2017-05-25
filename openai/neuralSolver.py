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
                 numEpisodesRun=10,
                 mode='cpu',
		 displayGraphics=False
                 ):
        
        self.gameEnv = gameEnv
        self.modelType = modelType
        self.epsilon = epsilon  # uses random action or predicted action based on epsilon value
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
        self.mode=mode
        self.agent = util.lookup(agentClass, globals())(actionFn=self.getLegalActions)
        self.displayGraphics = displayGraphics

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
            inputImageVector = tf.placeholder("float", [None, 80, 80, 4])
    
            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(inputImageVector, W_conv1, 4) + b_conv1)
            
            h_pool1 = self.max_pool_2x2(h_conv1)
            
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    
            # fc_out layer
            fc_out = tf.matmul(h_fc1, W_fc2) + b_fc2
            
            predictedActionScoreVector = tf.placeholder("float", [None, self.numActions])
        
            actualScore = tf.placeholder("float", [None])  # scalar value
            
            predictedScore = tf.reduce_sum(tf.multiply(fc_out, predictedActionScoreVector), axis=1)  # scalar
            
            cost = tf.reduce_mean(tf.square(actualScore - predictedScore))  # this is more of regression kind of cost function 
            
            tf.summary.scalar("loss", cost)  # Create predictedActionScoreVector summary to monitor cost tensor
            
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
                    merged_summary_op,
                    displayGraphics=True):
        
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
            self.agent.startEpisode() 
            print(displayGraphics)
            self.gameEnv.render(close=not displayGraphics)
            gameAction = random.choice(self.getLegalActions(initialColoredObservation))  # choose predictedActionScoreVector scalar randomly from predictedActionScoreVector set of legal actions
            
            initialColoredObservation, _, _, _ = self.gameEnv.step(gameAction)  # pass in scalar action to get output
            imageBackgroundGray = self.convertImageBackgroubtToGray(initialColoredObservation)  # do pre-processing on image
            currentLastFourImageFrames = np.stack((imageBackgroundGray, imageBackgroundGray, imageBackgroundGray, imageBackgroundGray), axis=2)
            
            done = False
            
            while not done :
                            
                yout_t = sess.run(fc_out,feed_dict={inputImageVector: [currentLastFourImageFrames]})  # similar to sess.run(y_out,feed_dict={X:x,is_training:True})
                actionVector = np.zeros([self.numActions])
                action_index = 0
                 
                if random.random() <= self.epsilon:  # we will gradually decrease epsilon as we explore 
                    action_index = random.randrange(self.numActions)
                    actionVector[random.randrange(self.numActions)] = 1
                else:
                    action_index = np.argmax(yout_t)
                    actionVector[action_index] = 1
                
                nextColoredImageObservation, reward, isEpisodeDone, _ = self.gameEnv.step(action_index)  # run the selected action and observe next state and reward
                
                nextImageBackgroundGray = cv2.cvtColor(cv2.resize(nextColoredImageObservation, (80, 80)), cv2.COLOR_BGR2GRAY)  # pre processing 
                _, nextImageBackgroundGray = cv2.threshold(nextImageBackgroundGray, 1, 255, cv2.THRESH_BINARY)
                nextImageBackgroundGray = np.reshape(nextImageBackgroundGray, (80, 80, 1))
                nextLastFourImageFrames = np.append(nextImageBackgroundGray, currentLastFourImageFrames[:, :, 0:3], axis=2)  # stack last 4 image frames 
                
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
                
                # save progress every 10000 iterations
                if numIterations % self.numStepsBeforeSaveModel == 0:
                    saver.save(sess, saved_networks_path + self.game + '-dqn', global_step=numIterations)
    
                # print info
                state = ""
                if numIterations <= self.numTraining:
                    state = "observe"
                elif numIterations > self.numTraining and numIterations <= self.numTraining + self.explore:
                    state = "explore"
                else:
                    state = "train"
    
                print("TIMESTEP", numIterations, "/ EPISODE", episodeNum, "/ STATE", state, \
                      "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                      "/ Q_MAX %e" % np.max(yout_t))

                if reward > 0:
                    print("GREAT SUCCESS! reward = ", reward) 

            # scale down epsilon as we train
            # this is predictedActionScoreVector linear decay self.epsilon -= self.epsilonDecay  / self.numEpisodesRun
            self.epsilon *= self.epsilonDecay
            if episodeNum > self.numEpisodesRun:
                self.epsilon = 0
        self.gameEnv.close()
  
    
    def playGame(self):
        #sess = tf.InteractiveSession()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op = self.createNetwork()
        self.trainNetwork(sess, inputImageVector, fc_out, predictedActionScoreVector, actualScore, cost, optimizer, merged_summary_op, self.displayGraphics)
        
    
    def convertImageBackgroubtToGray(self, currentImageOutputColored):
        imageWithBackgroundGray = cv2.cvtColor(cv2.resize(currentImageOutputColored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, imageWithBackgroundGray = cv2.threshold(imageWithBackgroundGray, 1, 255, cv2.THRESH_BINARY)
        return imageWithBackgroundGray    
    
    
        
    def main(self):
        self.playGame()
