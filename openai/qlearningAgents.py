from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        #BEGIN SOLUTION
        self.qVals = {}
        #END SOLUTION

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        #BEGIN SOLUTION METHOD
        stateQVals = self.qVals.setdefault(state,util.Counter())
        return stateQVals[action]
        #END SOLUTION

    #BEGIN SOLUTION NO PROMPT
    def getPolicyAndValue(self, state):
        bestAct, bestVal = None, None
        actions = self.getLegalActions(state)
        if len(actions) == 0: return (None, 0.0)
        for act in actions:
            val = self.getQValue(state,act)
            if bestVal == None or val > bestVal:
                bestAct, bestVal = act, val
        return bestAct, bestVal
    #END SOLUTION

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        #BEGIN SOLUTION METHOD
        return self.getPolicyAndValue(state)[1]
        #END SOLUTION

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        #BEGIN SOLUTION METHOD
        return self.getPolicyAndValue(state)[0]
        #END SOLUTION

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        #BEGIN SOLUTION METHOD
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        #END SOLUTION

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        #BEGIN SOLUTION METHOD
        # getQValue will ensure qVals[state][action] inited
        oldQVal = self.getQValue(state,action)
        newQVal = reward + self.discount * \
          self.computeValueFromQValues(nextState)
        self.qVals[state][action] += \
          self.alpha * (newQVal-oldQVal)
        #print 'state', state
        #print 'newState', nextState
        #print 'oldQ, newQ', oldQVal, newQVal
        #print len(self.qVals)
        #if oldQVal != 0:
        #    print "hello!"
        return
        #END SOLUTION

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        #BEGIN SOLUTION METHOD
        feats = self.featExtractor.getFeatures(state,action)
        qVal = 0.0
        for f,v in feats.items():
            qVal += self.weights[f] * v
        return qVal
        #END SOLUTION

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #BEGIN SOLUTION METHOD
        feats = self.featExtractor.getFeatures(state,action)
        oldQVal = self.getQValue(state,action)
        newQVal = reward + self.discount * self.computeValueFromQValues(nextState)
        update = self.alpha * (newQVal-oldQVal)
        for f,v in feats.items():
            self.weights[f] += update * v
        #END SOLUTION

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # BEGIN SOLUTION ALT="pass"
            if len(self.weights) < 20:
                print "Feature Weights"
                print self.weights
            # END SOLUTION

class MountainCarQAgent(ApproximateQAgent):
    def __init__(self, extractor='MountainCarExtractor', **args):
        ApproximateQAgent.__init__(self, extractor=extractor, **args)

class AcrobotQAgent(ApproximateQAgent):
    def __init__(self, extractor='AcrobotExtractor', **args):
        ApproximateQAgent.__init__(self, extractor=extractor, **args)

class FlappyBirdQAgent(ApproximateQAgent):
    def __init__(self, extractor='FlappyBirdExtractor', **args):
        ApproximateQAgent.__init__(self, extractor=extractor, **args)

class LunarLanderQAgent(ApproximateQAgent):
    def __init__(self, extractor='LunarLanderExtractor', **args):
        ApproximateQAgent.__init__(self, extractor=extractor, **args)
