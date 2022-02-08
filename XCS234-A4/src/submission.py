import numpy as np
import csv
import os
import pdb

from abc import ABC, abstractmethod
from utils.data_preprocessing import load_data, dose_class, LABEL_KEY


# Base classes
class BanditPolicy(ABC):
	@abstractmethod
	def choose(self, x): pass

	@abstractmethod
	def update(self, x, a, r): pass

class StaticPolicy(BanditPolicy):
	def update(self, x, a, r): pass

class RandomPolicy(StaticPolicy):
	def __init__(self, probs=None):
		self.probs = probs if probs is not None else [1./3., 1./3., 1./3.]

	def choose(self, x):
		return np.random.choice(('low', 'medium', 'high'), p=self.probs)

############################################################
# Problem 1: Estimation of Warfarin Dose
############################################################

############################################################
# Problem 1a: baselines

class FixedDosePolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x (dict): dictionary containing the possible patient features.

		Returns:
			output (str): string containing one of ('low', 'medium', 'high')

		TODO:
			Please implement the fixed dose which is tp assign medium dose 
			to all patients.
		"""
		### START CODE HERE ###
		return('medium')
		### END CODE HERE ###


class ClinicalDosingPolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x (dict): Dictionary containing the possible patient features. 

		Returns:
			output (str): string containing one of ('low', 'medium', 'high')

		TODO:
			- Prepare the features to be used in the clinical model 
			  (consult section 1f of appx.pdf for feature definitions)
			- Create a linear model based on the values in section 1f 
			  and return its output based on the input features

		Hint:
			- Look at the utils/data_preprocessing.py script to see the key values
			  of the features you can use. The age in decades is implemented for 
			  you as an example.
			- You can treat Unknown race as missing or mixed race.
			- Use dose_class() implemented for you. 
		"""
		age_in_decades = x['Age in decades']

		### START CODE HERE ###
		height = x['Height (cm)']
		weight = x['Weight (kg)']
		asian = x['Asian']
		african = x['Black']
		missing = x['Unknown race']
		carb = x['Carbamazepine (Tegretol)']
		phen = x['Phenytoin (Dilantin)']
		rif = x['Rifampin or Rifampicin']
		if carb > 0 or phen > 0:
			enzyme = 1
		elif rif > 0:
			enzyme = 1
		else:
			enzyme = 0
		amio = x['Amiodarone (Cordarone)']
		dose = (4.0376
				- 0.2546 * age_in_decades
				+ 0.0118 * height
				+ 0.0134 * weight
				- 0.6752 * asian
				+ 0.4060 * african
				+ 0.0443 * missing
				+ 1.2799 * enzyme
				- 0.5695 * amio
				)
		dose = dose**2
		return dose_class(dose)
		### END CODE HERE ###


############################################################
# Problem 1b: upper confidence bound linear bandit

class LinUCB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation" 

		Args:
			n_arms (int): the number of different arms/ actions the algorithm can take 
			features (list of str): contains the patient features to use 
			alpha (float): hyperparameter for step size. 
		
		TODO:
			- Please initialize the following internal variables for the Disjoint Linear UCB Bandit algorithm:
				* self.n_arms 
				* self.features
				* self.d
				* self.alpha
				* self.A
				* self.b
			  (these terms align with the paper, please refer to the paper to understadard what they are) 
			- Feel free to add additional internal variables if you need them, but they are not necessary. 

		Hint:
			Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
		"""
		### START CODE HERE ###
		self.n_arms = n_arms
		self.features = features
		self.d = len(self.features)
		self.alpha = alpha
		self.A = []
		self.b = []
		self.bm = []
		self.AI = []
		self.theta = []
		for _ in range(3):
			self.A.append(np.identity(self.d))
			self.b.append(np.zeros(self.d))
			self.bm.append(np.zeros((self.d, 1)))
			self.AI.append(np.identity(self.d))
			self.theta.append(np.zeros((self.d, 1)))
		### END CODE HERE ###


	def choose(self, x):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation" 

		Args:
			x (dict): Dictionary containing the possible patient features. 
		Returns:
			output (str): string containing one of ('low', 'medium', 'high')

		TODO:
			Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm. 
		"""
		### START CODE HERE ###
		xfv = [ x[self.features[indx]] for indx in range (self.d)]
		xtn = (np.array(xfv)).reshape(1,self.d)
		xt = xtn
		xtT = np.transpose(xt)
		at = np.zeros((1, len(self.A)))
		for a_idx in range(len(self.A)):
			pa = (np.matmul(np.transpose(self.theta[a_idx]), xtT) + self.alpha * np.sqrt(np.matmul(np.matmul(xt, self.AI[a_idx]), xtT)))
			at[0][a_idx] = float(pa)
			self.a_max = int(np.squeeze(np.argmax(at, axis=1)))
		if self.a_max == 0: dose = "low"
		elif self.a_max == 1:
			dose = "medium"
		else:
			dose = "high"
		return dose
		### END CODE HERE ###


	def update(self, x, a, r):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation" 

		Args:
			x (dict): Dictionary containing the possible patient features. 
			a (str): string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r (int): the reward you recieved for that action

		TODO:
			Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm. 

		Hint: 
			Which parameters should you update?
		"""
		### START CODE HERE ###
		### END CODE HERE ###


############################################################
# Problem 1c: eGreedy linear bandit

class eGreedyLinB(LinUCB):
	def __init__(self, n_arms, features, alpha=1.):
		super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.)
		self.time = 0  
	def choose(self, x):
		"""
		Args:
			x (dict): Dictionary containing the possible patient features. 
		Returns:
			output (str): string containing one of ('low', 'medium', 'high')

		TODO:
			Instead of using the Upper Confidence Bound to find which action to take, 
			compute the probability of each action using a simple dot product between Theta & the input features.
			Then use an epsilion greedy algorithm to choose the action. 
			Use the value of epsilon provided
		"""
		
		self.time += 1 
		epsilon = float(1./self.time)* self.alpha
		### START CODE HERE ###
		### END CODE HERE ###



############################################################
# Problem 1d: Thompson sampling

class ThomSampB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 

		Args:
			n_arms (int): the number of different arms/ actions the algorithm can take 
			features (list of str): contains the patient features to use 
			alpha (float): hyperparameter for step size.
		
		TODO:
			- Please initialize the following internal variables for the Thompson sampling bandit algorithm:
				* self.n_arms 
				* self.features
				* self.d
				* self.v2
				* self.B
				* self.mu
				*self.f
			  (these terms align with the paper, please refer to the paper to understadard what they are) 
			Please feel free to add additional internal variables if you need them, but they are not necessary. 

		Hints:
			- Keep track of a seperate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
			- Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm 
				based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
				values for the arm that we selected
			- What the paper refers to as b in our case is the medical features vector
			- The paper uses a summation (from time =0, .., t-1) to compute the model paramters at time step (t),
				however if you can't access prior data how might one store the result from the prior time steps.
		
		"""

		### START CODE HERE ###
		### END CODE HERE ###



	def choose(self, x):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 

		Args:
			x (dict): Dictionary containing the possible patient features. 
		Returns:
			output (str): string containing one of ('low', 'medium', 'high')

		TODO:
			- Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm. 
			- Please use the gaussian distribution like they do in the paper
		"""

		### START CODE HERE ###
		### END CODE HERE ###


	def update(self, x, a, r):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs" 
			
		Args:
			x (dict): Dictionary containing the possible patient features. 
			a (str): string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r (int): the reward you recieved for that action

		TODO:
			- Please implement the update step for Disjoint Thompson Sampling Bandit algorithm. 
			- Please use the gaussian distribution like they do in the paper

		Hint: 
			Which parameters should you update?
		"""

		### START CODE HERE ###
		### END CODE HERE ###
