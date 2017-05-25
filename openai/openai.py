import gym
from gym import wrappers
import gym_ple
import numpy as np
import util
import sys, time
from copy import copy
from neuralSolver import  NeuralSolver

# some default bin settings to get started with tabular qlearning
BIN_SIZES = {
    # cart_x, cart_velocity, pole_theta, pole_velocity
    'CartPole-v1': [5, 10, 20, 10],
    # car_x, car_velocity
    'MountainCar-v0': [5, 5],
    #
    'Pendulum-v0': [5, 5, 5],
    #
    'Acrobot-v1': [5, 5, 5, 5, 10, 10],
}

BIN_RANGES = {
    'CartPole-v1': ((-2.4, 2.4), (-2.0, 2.0), (-.2, .2), (-3.0, 3.0)),
}

def runNeuralSolver(env_name,
                    num_bins,
                    alpha,
                    epsilon,
                    gamma,
                    alpha_decay,
                    epsilon_decay,
                    numTraining,
                    display_graphics,
                    save_recording,
                    agent_class,
                    discretize,
                    sim_env,
                    num_actions,
                    num_steps_save,
                    num_episodes_run,
                    mode
                    ):
    
    env = gym.make(env_name)
    recording_str = 'recording/' + env_name + '-' + time.strftime('%H%M%S')
    if save_recording:
        save_recording=None #can be some function called for recording video
    env = wrappers.Monitor(env, recording_str, video_callable=save_recording)
    neuralSolver = NeuralSolver(gameEnv=env,
                                agentClass=agent_class,
                                epsilon=epsilon,
                                epsilonDecay=epsilon_decay,
                                gamma=gamma,
                                numTraining=numTraining,
                                learningRate=alpha,
                                learningDecay=alpha_decay,
                                numActions=num_actions,
                                numStepsBeforeSaveModel=num_steps_save,
                                numEpisodesRun=num_episodes_run,
                                mode=mode,
				                displayGraphics=display_graphics
                                )
    
    neuralSolver.playGame()  # add boolean flag for train or test
    
    gym.scoreboard.api_key = "sk_KFRYZyR1QO2HdihQOHsljA" 
    gym.upload(recording_str)

def runOpenAi(env_name, num_bins, alpha, epsilon, gamma, alpha_decay, epsilon_decay, numTraining, display_graphics, agent_class, discretize, sim_env, learning_rate, num_actions, num_steps_save, num_episodes_run,mode):
    env = gym.make(env_name)
    recording_str = 'recording/' + env_name + '-' + time.strftime('%H%M%S')
    env = wrappers.Monitor(env, recording_str)
    openai_learner = OpenAiLearner(env, env_name, num_bins, alpha, epsilon, gamma, alpha_decay, epsilon_decay, numTraining, agent_class, discretize, sim_env, learning_rate, num_actions, num_steps_save, num_episodes_run,mode)
    
    for i_episode in range(openai_learner.numTraining + 100):
        observation = env.reset()
        state = openai_learner.prepareState(observation)
        
        old_state = None
        openai_learner.agent.startEpisode()
        done = False
        env.render(close=not display_graphics)
        
        while not done:
            if display_graphics:
                env.render()
            
            action = openai_learner.getAction(state)
            old_state = state
            observation, reward, done, info = env.step(action)
            state = openai_learner.prepareState(observation)
            openai_learner.observeTransition(old_state, action, state, reward)
        
        openai_learner.agent.stopEpisode()
        openai_learner.updateParameters()
        if i_episode % (openai_learner.numTraining / 10.) > (i_episode + 1) % (openai_learner.numTraining / 10.):
            pass
            # print('episode:', i_episode, '# of observed states', len(openai_learner.agent.qVals) )
            # print 'weights:', openai_learner.agent.weights
            # print 'epsilon:', openai_learner.epsilon
            # print 'alpha:', openai_learner.alpha
    
    env.close()
    gym.scoreboard.api_key = "sk_KFRYZyR1QO2HdihQOHsljA" 
    gym.upload(recording_str)
 
class OpenAiLearner:

    def __init__(self, env, env_name, num_bins, alpha, epsilon, gamma, alpha_decay, epsilon_decay, numTraining, agent_class, discretize, sim_env, learning_rate, num_actions, num_steps_save, num_episodes_run,mode):
        self.env = env
        self.env_name = env_name

        self.numTraining = numTraining
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.discretize = discretize
        self.sim_env = sim_env
        
        self.agent = util.lookup(agent_class, globals())(actionFn=self.getLegalActions,
          numTraining=self.numTraining,
          epsilon=self.epsilon,
          alpha=self.alpha,
          gamma=self.gamma)
        
        if self.discretize:
            self.discretizeSpace(num_bins)
    
    def getAction(self, observation):
        return self.agent.getAction(observation)

    def getLegalActions(self, state=None):
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            return tuple(range(self.env.action_space.n))

        # if isinstance(self.env.action_space, gym.spaces.box.Box):
        #    low = self.env.action_space.low[0]
        #    high = self.env.action_space.high[0]
        #    actions = tuple(np.linspace(low, high, 10))
        #    return [actions]

        raise Exception('Action type not supported: %s' % type(self.env.action_space))

    def observeTransition(self, old_obs, action, observation, reward):
        self.agent.observeTransition(old_obs, action, observation, reward)

    def updateParameters(self):
        self.epsilon *= self.epsilon_decay
        self.alpha *= self.alpha_decay
        # self.agent.setEpsilon(self.epsilon)

    def prepareState(self, observation):
        
        print('prepareState Observation', observation.shape)
        if isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
            pass
        
        elif isinstance(self.env.observation_space, gym.spaces.box.Box):
            if self.discretize:
                observation = self.toBins(observation)
            observation = tuple(observation)
        
        else:
            raise Exception('Observation type not supported: %s' % type(self.env.observation_space))
    
        if self.sim_env:
            observation = (copy(self.env), observation)
        return observation

    def discretizeSpace(self, num_bins):
        self.bins = []
        if isinstance(self.env.observation_space, gym.spaces.box.Box):
            bin_ranges = zip(self.env.observation_space.low, self.env.observation_space.high)
            if self.env_name in BIN_RANGES:
                bin_ranges = BIN_RANGES[self.env_name]
            if len(bin_ranges) != len(num_bins):
                print('Incorrect number of bins specified:', len(num_bins), 'Should be:', len(bin_ranges), '\nFalling back to default')
                num_bins = BIN_SIZES[self.env_name]
            i = 0
            for low, high in bin_ranges:
                self.bins.append(np.linspace(low, high, num_bins[i]))
                i += 1

    def toBins(self, obs):
        obs = list(obs)
        binned_obs = []
        for i in range(len(obs)):
            bin_type = self.bins[i]
            binned_obs.append(np.digitize(x=[obs[i]], bins=bin_type)[0])
        return tuple(binned_obs)

def read_command(argv):
    """
    Processes the command used to run openai gym from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python openai.py <options>
    EXAMPLES:   (1) python openai.py
                    - starts training on default environment and learning parameters
                (2) python openai.py -v MountainCar-v0 -b 30,30
                    - starts training on mountain car environment with bucket sizes for each of the (2) state space dimensions as 30
                (3) python openai.py --no-graphics
                    - starts training without graphics.
                (4) sudo xvfb-run -s "-screen 0 1400x900x24" python openai.py -x 50 -f deepLearningAgent.DeepLearningAgent -v FlappyBird-v0
                    - starts training a deep learning agent running off a server
    """
    parser = OptionParser(usageStr)
    
    parser.add_option('-v', '--environment', type='string', dest='env_name', help=default('Environment Name'),
                        default='CartPole-v1')
   
    # we don't need this 
    parser.add_option('-b', '--bins', type='string', action='callback', callback=parse_comma_separated_args, dest='num_bins', help=default('an array of bin count values for each dimension \'bin1_size, ... , binN_size\''),
                        default=[5, 10, 20, 10])
    
    parser.add_option('-a', '--alpha', type='float', dest='alpha', help=default('Learning rate'),
                        default='1e-6')
    
    parser.add_option('-e', '--epsilon', type='float', dest='epsilon', help=default('Exploration rate'),
                        default='1.0')
    
    parser.add_option('-g', '--gamma', type='float', dest='gamma', help=default('Discount factor'),
                        default='0.997')
    
    # we don't need this yet ( may be later)
    parser.add_option('--learningDecay', type='float', dest='alpha_decay', help=default('Learning rate decay factor'),
                        default='1.0')
    
    parser.add_option('--epsilonDecay', type='float', dest='epsilon_decay', help=default('Exploration rate decay factor'),
                        default='.01')
    
    parser.add_option('-x', '--numTraining', type='int', dest='numTraining', help=default('Number of iterations before we start training'),
                        default='50000')
    parser.add_option('--graphics', action='store_true', dest='display_graphics', help=default('Enable graphic display'),
                        default=True)
    parser.add_option('--no-graphics', action='store_false', dest='display_graphics', help=default('Disable graphic display'),
                        default=True)
    parser.add_option('--no-recording', action='store_false', dest='save_recording', help=default('Disable episode recording'),
                        default=True)
    # we dont need this 
    parser.add_option('--discretize', action='store_true', dest='discretize', help=default('Discretize the state space into bins. Specify bin sizes with -b'),
                        default=False)
    # we dont need this 
    parser.add_option('--simulateEnv', action='store_true', dest='sim_env', help=default('Include a copy of the environment as part of the state in the form (env, state)'),
                        default=False)
    
    parser.add_option('-f', '--agent', type='string', dest='agent_class', help=default('Agent class name'),
                        default='qlearningAgents.QLearningAgent')
    
    parser.add_option('--numActions', type='int', dest='num_actions', help=default('Number of valid actions'),
                        default='2')
    
    parser.add_option('--numStepsBeforeSaveModel', type='int', dest='num_steps_save', help=default('Number of steps before we save the model for checkpoint'),
                        default='10000')
    
    parser.add_option('--numEpisodesRun', type='int', dest='num_episodes_run', help=default('Number of episodes before we terminate'),
                        default='15')
    
    parser.add_option('--runMode', type='string', dest='mode', help=default('GPU or CPU , if you want GPU mode give option gpu otherwise defaults to cpu'),default='cpu')
    
    options, otherjunk = parser.parse_args(argv)
    assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
    return vars(options)

def default(str):
    return str + ' [Default: %default]'

def parse_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
    options = read_command(sys.argv[1:])
    # runOpenAi(**options)
    runNeuralSolver(**options)
