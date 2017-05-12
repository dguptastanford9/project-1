"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math
from copy import copy

class FeatureExtractor:  


    
    def getFeatures(self, state, action):        
        """
            Returns a dict from features to counts
            Usually, the count will just be 1.0 for
            indicator functions.    
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """
    
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()
        
        features["bias"] = 1.0
        
        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height) 
        features.divideAll(10.0)
        return features

class MountainCarExtractor(FeatureExtractor):
    
    # action 0, 1, 2
    # -x dir, none, +x dir
    def getFeatures(self, state, action):
        features = util.Counter()
        
        
        x, v = state
        pos_v = v > 0
        neg_v = v > 0
        pos_a = action == 2
        neg_a = action == 0
        dir_vel = (pos_v and pos_a) or (neg_v and neg_a)
        features['direction-of-velocity'] = 1 if dir_vel else 0
        
        #features['velocity-magnitude'] = math.fabs(v)

        features['near-edge'] = 1 if x < -1.0 else 0
        
        features['stop-action'] = 1 if action == 1 else 0

        features[int(x*10.)/10.] = 1.
        
        return features

class AcrobotExtractor(FeatureExtractor):

    # action 0, 1, 2
    # -torque, none, +torque
    def getFeatures(self, state, action):
       x1, y1, x2, y2, t1, t2 = state

       #if x1 < .5:
       #    print state
       
       features = util.Counter() 
       
       #theta1 = math.acos(x1)
       #theta2 = math.acos(x2)
       #features['x1_val'] = x1
       #features['x1 + y1'] = math.cos(theta1+theta2)
       #features['x2_val'] = x2
       #features['y1_val'] = y1
       #features['y2_val'] = y2*y2
       
       features['t1_val'] = t1 / (4 * 3.14) * (action-1)
       features['t2_val'] = t2 / (9 * 3.14) * (action-1)

       return features

class FlappyBirdExtractor(FeatureExtractor):

    # action 0, 1
    # nothing, flap up
    #   
    # observation space: 512 x 288 image array of (r, g, b) values
    def getFeatures(self, state, action):

        features = util.Counter()



        return features


def applyAction(environment, action):
    env = copy(environment)
    env._monitor = copy(env._monitor)
    env._monitor.video_recorder = None
    #del env._monitor

    observation, reward, done, info = env.step(action)
    return observation

class LunarLanderExtractor(FeatureExtractor):

    # action 0, 1, 2, 3
    # nothing, left engine, main engine, right engine
    #   
    # observation space: 
    # x_pos, y_pos, x_vel, y_vel, angle, rotation, left_leg, right_leg
    def getFeatures(self, state, action):

        env, obs = state
        x_pos = obs[0]
        y_pos = obs[1]
        x_vel = obs[2]
        y_vel = obs[3]
        angle = obs[4]
        rotation = obs[5]
        features = util.Counter()
        #print angle, rotation

        nextState = applyAction(env, action)

        #features['x_offset'] = 1. if math.fabs(x_pos) > .1 else 0.
        #features['x_vel_mag'] = math.fabs(x_vel)
        #features['y_vel_mag'] = math.fabs(y_vel)
        #features['dist_from_goal'] = math.sqrt(x_pos*x_pos + y_pos*y_pos)

        thresh = .05

        features['slow_down'] = 0.
        if y_vel > thresh:
            if action == 0:
                features['slow_down'] = 1.
        elif y_vel < -thresh:
            if action == 2:
                features['slow_down'] = 1.
       
        features['straighten'] = 0.
        features['recenter'] = 0.
        if angle > thresh:
            if x_pos < -thresh and action == 2:
                features['recenter'] = 1.
            elif x_pos > thresh and action == 3:
                features['straighten'] = 1.
        elif angle < -thresh:
            if x_pos > thresh and action == 2:
                features['recenter'] = 1.
            elif x_pos < -thresh and action == 1:
                features['straighten'] = 1.

        features.divideAll(10)
        return features
