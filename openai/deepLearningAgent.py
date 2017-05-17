import numpy as np
import random
 

class DeepLearningAgent(object):

    def __init__(self, **args):
        self.actionFn = args['actionFn'] 

    def getAction(self, observation):
        """  
        called by environment for every observation
        """
        actions = self.getLegalActions(observation)
        print('actions',actions)
        pixels = np.array(observation) # numpy array of current image, W x H x 3 (r,g,b)
        print('random.choice(actions)',random.choice(actions))
        return random.choice(actions)
    
    
        
    
    def observeTransition(self, old_obs, action, observation, reward):
        """
        called by environment between two observations
        """
        pass

    def startEpisode(self):
        """
        Called by environment when new episode is starting
        """

        pass

    def stopEpisode(self):
        """
        Called by environment when episode is done
        """
        pass

    def getLegalActions(self, state):
        """
        input:
            state: an observation from the environment
        output:
            list of legal actions
        """
        return self.actionFn(state)
