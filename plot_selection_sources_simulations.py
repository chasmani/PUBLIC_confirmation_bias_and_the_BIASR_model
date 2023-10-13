

import numpy as np

PROB_TRUE_R = 0.75
PROB_TRUE_NOT_R = 0.5


def get_joint_prior_two_sources(prior_H, prior_R_a, prior_R_b):

	joint_prior_tensor = np.empty(shape=(2,2,2))
	
	p_H = [1-prior_H, prior_H]
	p_R_a = [1-prior_R_a, prior_R_a]
	p_R_b = [1-prior_R_b, prior_R_b]

	for H in [0,1]:
		for R_a in [0,1]:
			for R_b in [0,1]:
				joint_prior_tensor[H, R_a, R_b] = p_H[H] * p_R_a[R_a] * p_R_b[R_b]

	return joint_prior_tensor



def get_prob_D_given_H_R(H, R, D):

	if R == 1:
		if D == H:
			return PROB_TRUE_R
		else:
			return 1 - PROB_TRUE_R
	elif R == 0:
		if D == H:
			return PROB_TRUE_NOT_R
		else:
			return 1 - PROB_TRUE_NOT_R


def get_likelihood_tensor(D, D_from="B"):
	"""
	Probabiltiy of D given H and R_a and R_b
	"""

	joint_likelihood_tensor = np.empty(shape=(2,2,2))

	for H in [0,1]:
		for R_a in [0,1]:
			for R_b in [0,1]:
				if D_from == "A":
					joint_likelihood_tensor[H, R_a, R_b] = get_prob_D_given_H_R(H, R_a, D)
				elif D_from == "B":
					joint_likelihood_tensor[H, R_a, R_b] = get_prob_D_given_H_R(H, R_b, D)

	return joint_likelihood_tensor

def get_likelihood_simple(D, prior_R):

	joint_likelihood_tensor = np.empty(shape=(2,2,2))
	for H in [0,1]:
		if D == H:
			likelihood_H = PROB_TRUE_R * prior_R + PROB_TRUE_NOT_R * (1-prior_R)
		else:
			likelihood_H = (1-PROB_TRUE_R) * prior_R + (1-PROB_TRUE_NOT_R) * (1-prior_R)
		for R_a in [0,1]:
			for R_b in [0,1]:
				joint_likelihood_tensor[H, R_a, R_b] = likelihood_H
		
	return joint_likelihood_tensor


def get_posterior_rational(prior, D, source):

	unnormed_posterior = get_unnormed_posterior_rational(prior, D, source)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_unnormed_posterior_rational(prior, D, source):
	
	likelihood = get_likelihood_tensor(D=D, D_from=source)
	return np.multiply(prior, likelihood)


def get_posterior_indy(prior, D, source):

	unnormed_posterior = get_unnormed_posterior_indy(prior, D, source)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_unnormed_posterior_indy(prior, D, source):

	# Perform the independence approximation
	prob_H = np.sum(prior, axis=(1,2))[1]
	prob_R_a = np.sum(prior, axis=(0,2))[1]
	prob_R_b = np.sum(prior, axis=(0,1))[1]

	prior = get_joint_prior_two_sources(prob_H, prob_R_a, prob_R_b)

	likelihood = get_likelihood_tensor(D=D, D_from=source)

	return np.multiply(prior, likelihood)

def get_posterior_simple(prior, D, prob_R):

	unnormed_posterior = get_unnormed_posterior_simple(prior, D, prob_R)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_unnormed_posterior_simple(prior, D, prob_R):
	
	likelihood = get_likelihood_simple(D=D, prior_R = prob_R)
	# Get prob_H and prob_Rs
	return np.multiply(prior, likelihood)

def get_prob_D(D, prior, likelihood):

	unnormed_posterior = np.multiply(prior, likelihood)
	return np.sum(unnormed_posterior)

def replicate_biased_evaluation_and_assimilation():
	"""
	Replicate the results from one source using the two source machinery
	"""

	prob_H = 0.8
	prob_R = 0.5

	rational_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	indy_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	simple_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)

	Ds = [1,1,1,1,1]
	
	print(np.sum(rational_joint_prob, axis=2))

	for D in Ds:

		rational_joint_prob = get_posterior_rational(rational_joint_prob, D, source="A")
		print("\nData received")

		joint_prob_H_R_rat = np.sum(rational_joint_prob, axis=2)
		print("Rational")
		print(joint_prob_H_R_rat)

		indy_joint_prob = get_posterior_indy(indy_joint_prob, D, source="A")

		joint_prob_H_R_indy = np.sum(indy_joint_prob, axis=2)
		print("Indy")
		print(joint_prob_H_R_indy)

		simple_joint_prob = get_posterior_simple(simple_joint_prob, D=D, prob_R=prob_R)
		joint_prob_H_R_simple = np.sum(simple_joint_prob, axis=2)
		print("Simple")
		print(joint_prob_H_R_simple)		



def update_with_for_and_against():

	pass




if __name__=="__main__":
	#print(get_joint_prior_two_sources(0.5, 0.2,0.8))
	replicate_biased_evaluation_and_assimilation()