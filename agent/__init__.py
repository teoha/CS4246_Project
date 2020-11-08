try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
# try: # server-compatible import statement
#     from models import *
# except: pass
# try: # local-compatible import statement
#     from .models import *
# except: pass

import torch
from mcts import *
from dqn import *
from env import *
from models import AtariDQN
import torch.optim as optim
import numpy as np

env = construct_random_lane_env()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')
train_steps = 10 #dqn training steps
T = 1
RANDOM_SEED = 1234
max_iterations = 100 #dagger iteration
numiters = 100 #mcts iterations

# Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def simulatePolicy(state, model, env):
    '''
    Using apprentice model in MCTS simulation for playout.
    '''
    reward = 0.
    states = [GridWorldState(state=deepcopy(state), isDone=isDone)]
    isDone = False
    while not isDone:
        # Generate next best action with apprentice model
        tensor_state = torch.FloatTensor(state)
        model_state = torch.unsqueeze(tensor_state, 0)
        actions_q = model.forward(model_state)
        max_q, action = torch.max(actions_q[0], 0)
        # Transform state to GridWorldState
        next_state, done, reward, info = env.step(action.item())
        isDone = done
        grid_state = GridWorldState(state=deepcopy(next_state), is_done=isDone, reward=reward,
                                    prevState=deepcopy(state))
        states.append(grid_state)
        # transform next_state to torch!
        next_state_tensor = torch.FloatTensor(next_state)
        state_tensor = torch.FloatTensor(state)
        next_state = torch.cat([next_state_tensor, state_tensor[0:1, :, :]], 0)
        state = next_state
    return states


class ExampleAgent(Agent):
    '''
    An example agent that just output a random action.
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''
        # Initialize MCTS variables
        # TODO: Tune variables
        global env

        # Dagger variables
        self.model = AtariDQN((5,10,50), env.action_space.n).to(device)  # Torch default weights (can be any policy)

        # Initialize mcts simulator with the apprentice model as the playout policy
        self.mcts = MonteCarloTreeSearch(env=env, numiters=numiters, explorationParam=1.,random_seed=RANDOM_SEED, model=self.model)

        # Initialize empty prev state
        self.prevState = np.zeros((1,10,50))

    def reset(self, state, *args, **kwargs):
        ''' 
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        # Reset speed range for each lane
        

    def step(self, state, *args, **kwargs):
        ''' 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''
        # Initialize Dagger
        D = []
        # print(state.shape)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # curr_state = GridWorldState(state=newState, is_done=False)

        #Execute dagger
        for i in range(max_iterations):
            # TODO implement probability beta to use current policy

            # Run different trajetories
            states = simulatePolicy(state, self.model, env) # States traversed by rollout using current policy

            # Run MCTS to get best action for each state s
            for s in states:
                done = False
                gw_state = GridWorldState(state=s, is_done=done)
                q_values = self.mcts.buildTreeAndReturnQValues(initialState=gw_state)
                D.append(state, q_values)

            # TODO: Train the dqn with D (reward is estimated q for (s,action))
            # delta = Q(s,a) - d where d is element of D
            loss = optimize(self.model, D, optimizer)


    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''

        # TODO: Update current state
        # Step: state ->(position,...) -> state
        # Update: state -> (new position,....): speedrange -> state

        # self.prevState = np.expand_dims(state[0], 0)


def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''

    return ExampleAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_random_lane_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
        agent.initialize(**agent_init)
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)
