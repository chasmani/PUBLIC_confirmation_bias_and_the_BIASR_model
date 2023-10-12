

import numpy as np

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



def get_prob_D_given_H_R(H, R, D, prob_true_R, prob_true_not_R):

	if R == 1:
		if D == H:
			return prob_true_R
		else:
			return 1 - prob_true_R
	elif R == 0:
		if D == H:
			return prob_true_not_R
		else:
			return 1 - prob_true_not_R


def get_likelihood_tensor(prob_true_R, prob_true_not_R, D, D_from="B"):
	"""
	Probabiltiy of D given H and R_a and R_b
	"""

	joint_likelihood_tensor = np.empty(shape=(2,2,2))

	for H in [0,1]:
		for R_a in [0,1]:
			for R_b in [0,1]:
				if D_from == "A":
					joint_likelihood_tensor[H, R_a, R_b] = get_prob_D_given_H_R(H, R_a, D, prob_true_R, prob_true_not_R)
				elif D_from == "B":
					joint_likelihood_tensor[H, R_a, R_b] = get_prob_D_given_H_R(H, R_b, D, prob_true_R, prob_true_not_R)

	return joint_likelihood_tensor

def get_likelihood_simple(prob_true_R, prob_true_not_R, D, prior_R):

	joint_likelihood_tensor = np.empty(shape=(2,2,2))
	for H in [0,1]:
		if D == H:
			likelihood_H = prob_true_R * prior_R + prob_true_not_R * (1-prior_R)
		else:
			likelihood_H = (1-prob_true_R) * prior_R + (1-prob_true_not_R) * (1-prior_R)
		for R_a in [0,1]:
			for R_b in [0,1]:
				joint_likelihood_tensor[H, R_a, R_b] = likelihood_H
		
	return joint_likelihood_tensor


def get_posterior(prior, likelihood):

	unnormed_posterior = np.multiply(prior, likelihood)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_posterior_indy(prior, likelihood):


	# Perform the independence approximation
	prob_H = np.sum(prior, axis=(1,2))[1]
	prob_R_a = np.sum(prior, axis=(0,2))[1]
	prob_R_b = np.sum(prior, axis=(0,1))[1]

	prior = get_joint_prior_two_sources(prob_H, prob_R_a, prob_R_b)

	unnormed_posterior = np.multiply(prior, likelihood)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_posterior_simple(prior, likelihood):

	# Get prob_H and prob_Rs
	unnormed_posterior_prob_H = np.multiply(prior, likelihood)
	print(unnormed_posterior_prob_H)
	return unnormed_posterior_prob_H/np.sum(unnormed_posterior_prob_H)


def get_prob_D(D, prior, likelihood):

	unnormed_posterior = np.multiply(prior, likelihood)
	return np.sum(unnormed_posterior)

def replicate_biased_evaluation_and_assimilation():
	"""
	Replicate the results from one source using the two source machinery
	"""

	prob_H = 0.8
	prob_R = 0.5
	prob_true_R = 0.75
	prob_true_not_R = 0.5

	rational_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	indy_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	simple_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)



	Ds = [1,1,1,1,1]
	
	print(np.sum(rational_joint_prob, axis=2))

	for D in Ds:
		likelihood = get_likelihood_tensor(prob_true_R, prob_true_not_R, D=D, D_from="A")
		rational_joint_prob = get_posterior(rational_joint_prob, likelihood)
		print("\nData received")

		joint_prob_H_R_rat = np.sum(rational_joint_prob, axis=2)
		print("Rational")
		print(joint_prob_H_R_rat)

		indy_joint_prob = get_posterior_indy(indy_joint_prob, likelihood)

		joint_prob_H_R_indy = np.sum(indy_joint_prob, axis=2)
		print("Indy")
		print(joint_prob_H_R_indy)


		likelihood_simple = get_likelihood_simple(prob_true_R, prob_true_not_R, D=D, prior_R = prob_R)
		simple_joint_prob = get_posterior_simple(simple_joint_prob, likelihood_simple)
		joint_prob_H_R_simple = np.sum(simple_joint_prob, axis=2)
		print("Simple")
		print(joint_prob_H_R_simple)		




if __name__=="__main__":
	#print(get_joint_prior_two_sources(0.5, 0.2,0.8))
	replicate_biased_evaluation_and_assimilation()