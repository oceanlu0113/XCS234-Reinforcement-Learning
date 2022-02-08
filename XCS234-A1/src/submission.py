import argparse
import numpy as np
import gym
import time
from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='A program to run assignment 1 implementations.',
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--env", 
					help="The name of the environment to run your algorithm on.", 
					choices=["Deterministic-4x4-FrozenLake-v0","Stochastic-4x4-FrozenLake-v0"],
					default="Deterministic-4x4-FrozenLake-v0")

parser.add_argument("--algorithm", 
					help="The name of the algorithm to run.", 
					choices=["both","policy_iteration","value_iteration"],
					default="both")

# Register custom gym environments
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
	if 'Deterministic-4x4-FrozenLake-v0' in env:
		del gym.envs.registration.registry.env_specs[env]

	elif 'Deterministic-8x8-FrozenLake-v0' in env:
		del gym.envs.registration.registry.env_specs[env]

	elif 'Stochastic-4x4-FrozenLake-v0' in env:
		del gym.envs.registration.registry.env_specs[env]

	elif 'Stochastic-8x8-FrozenLake-v0' in env:
		del gym.envs.registration.registry.env_specs[env]

register(
	id='Deterministic-4x4-FrozenLake-v0',
	entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
	kwargs={'map_name': '4x4',
			'is_slippery': False})

register(
	id='Deterministic-8x8-FrozenLake-v0',
	entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
	kwargs={'map_name': '8x8',
			'is_slippery': False})

register(
	id='Stochastic-4x4-FrozenLake-v0',
	entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
	kwargs={'map_name': '4x4',
			'is_slippery': True})

register(
	id='Stochastic-8x8-FrozenLake-v0',
	entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
	kwargs={'map_name': '8x8',
			'is_slippery': True})

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P (dict): From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS (int): number of states in the environment
	nA (int): number of actions in the environment
	gamma (float): Discount factor. Number in range [0, 1)
"""

############################################################
# Problem 4: Frozen Lake MDP
############################################################

############################################################
# Problem 4a: policy evaluation

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""
	Evaluate the value function from a given policy.

	Args:
		P, nS, nA, gamma: defined at beginning of file
		policy (np.array[nS]): The policy to evaluate. Maps states to actions.
		tol (float): Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	
	Returns:
		value_function (np.ndarray[nS]): The value function of the given policy, where value_function[s] is
			the value of state s.
	"""

	value_function = np.zeros(nS)

	### START CODE HERE ###
	while True:
		# continue until max |value_function(s) - prev_value_function(s)| < tol
		terminate = 0.0
		# the policy to evaluate
		for state in range(nS):
			value = value_function[state]
			# Map states to actions
			evaluatePolicy = policy[state]
			# value_function[s] is the value of state s
			value_function[state] = sum([prob * (rew + gamma * value_function[nextState]) for (prob, nextState, rew, _) in P[state][evaluatePolicy]])
			# max |value_function(s) - prev_value_function(s)|
			terminate = max(terminate, abs(value - value_function[state]))
		# Terminate policy evaluation
		if terminate < tol:
			break
	### END CODE HERE ###
	return value_function


############################################################
# Problem 4b: policy improvement

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""
	Given the value function from policy improve the policy.

	Args:
		P, nS, nA, gamma: defined at beginning of file
		value_from_policy (np.ndarray): The value calculated from the policy
		policy (np.array): The previous policy

	Returns:
		new_policy (np.ndarray[nS]): An array of integers. Each integer is the optimal 
		action to take in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')

	### START CODE HERE ###
	# iterate through number of states
	for state in range(nS):
		optimalAction = None
		optimal = -float('inf')
		# iterate through number of actions
		for action in range(nA):
			# Each integer is the optimal action to take in that state according to the environment dynamics and the given value function
			current = sum([prob * (rew + gamma * value_from_policy[nextState]) for (prob, nextState, rew, _) in P[state][action]])
			# if the current integer is the optimal action to take in that state
			if current > optimal:
				optimalAction = action
				optimal = current
		new_policy[state] = optimalAction
	### END CODE HERE ###
	return new_policy


############################################################
# Problem 4c: policy iteration

def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	Args:
		P, nS, nA, gamma: defined at beginning of file
		tol (float): tol parameter used in policy_evaluation()
	
	Returns:
		value_function (np.ndarray[nS]): value function resulting from policy iteration
		policy (np.ndarray[nS]): policy resulting from policy iteration

	Hint: 
		You should call the policy_evaluation() and policy_improvement() methods to
		implement this method.
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	### START CODE HERE ###
	while True:
		# value function resulting from policy iteration
		value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
		# policy resulting from policy iteration
		newPolicy = policy_improvement(P, nS, nA, value_function, policy, gamma)
		# if all array elements are true, break
		if (newPolicy == policy).all():
			break
		policy = newPolicy
	### END CODE HERE ###
	return value_function, policy


############################################################
# Problem 4d: value iteration

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment

	Args:
		P, nS, nA, gamma: defined at beginning of file
		tol (float): Terminate value iteration when
				max |value_function(s) - prev_value_function(s)| < tol
	
	Returns:
		value_function (np.ndarray[nS]): value function resulting from value iteration
		policy (np.ndarray[nS]): policy resulting from value iteration

	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	### START CODE HERE ###
	while True:
		# continue until max |value_function(s) - prev_value_function(s)| < tol
		terminate = 0
		# iterate through number of states
		for state in range(nS):
			value = value_function[state]
			optimal = -float('inf')
			# iterate through number of actions
			for action in range(nA):
				# each integer is the optimal action to take in that state
				current = sum([prob * (rew + gamma * value_function[nextState]) for (prob, nextState, rew, _) in P[state][action]])
				optimal = max(optimal, current)
			# value function resulting from value iteration
			value_function[state] = optimal
			# max |value_function(s) - prev_value_function(s)| < tol
			terminate = max(terminate, abs(value - value_function[state]))
		# terminate value iteration
		if terminate < tol:
			break
	policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
	### END CODE HERE ###
	return value_function, policy

def render_single(env, policy, max_steps=100):
	"""
	This function does not need to be modified
	Renders policy once on environment. Watch your agent play!

	Args:
		env (gym.core.Environment): Environment to play on. Must have nS, nA, and P as
		  attributes.
		Policy (np.array[env.nS]): The action to take at a given state
	"""
	episode_reward = 0
	ob = env.reset()
	for t in range(max_steps):
		env.render()
		time.sleep(0.25)
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	env.render();
	if not done:
		print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
	else:
		print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
	# read in script arguments
	args = parser.parse_args()
	
	# Make gym environment
	env = gym.make(args.env)

	if (args.algorithm == "both") | (args.algorithm == "policy_iteration"):
		print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

		V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
		render_single(env, p_pi, 100)

	if (args.algorithm == "both") | (args.algorithm == "value_iteration"):

		print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

		V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
		render_single(env, p_vi, 100)
