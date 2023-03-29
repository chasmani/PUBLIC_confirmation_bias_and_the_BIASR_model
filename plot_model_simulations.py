
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


COLOR_INDY = "#2980b9"
COLOR_RATIONAL = "#e67e22"
COLOR_SIMPLE = "#6ab04c"
MARKER_INDY = "o"
MARKER_RATIONAL = "s"
MARKER_SIMPLE = "x"
LINESTYLE_INDY = "solid"
LINESTYLE_SIMPLE = "dashed"
LINESTYLE_RATIONAL = "dotted"

LINEWIDTH_FOR = 2
LINEWIDTH_AGAINST = 1

SIZE_SOURCES = 10

def get_joint_prior_matrix(prior_R, prior_H):
	"""
	Given priors in P(R) and P(H)
	Get a joint belief matrix for P(H,R)
	"""
	joint_prior_matrix = np.array([
		[prior_H*prior_R, (1-prior_H)*prior_R],
		[prior_H*(1-prior_R), (1-prior_H)*(1-prior_R)],
		])
	return joint_prior_matrix
	

def get_joint_prob_matrix_d_given_h(prob_true_R, prob_true_not_R, X):
	"""
	Get the probabilty of data, P(D|H,R) in matrix form
	"""
	if X == 1:
		joint_prob_d_given_h = np.array([
			[prob_true_R, 1-prob_true_R],
			[prob_true_not_R, 1-prob_true_not_R]
			])

	if X == 0:
		joint_prob_d_given_h = np.array([
			[1-prob_true_R, prob_true_R],
			[1-prob_true_not_R, prob_true_not_R]
			])	

	return joint_prob_d_given_h


def get_joint_posterior(joint_prob_matrix, joint_prob_d_given_h):
	"""
	Get the rational posterior matrix given some data
	"""
	unnormed_posterior = np.multiply(joint_prob_matrix, joint_prob_d_given_h)
	posteriors = unnormed_posterior / np.sum(unnormed_posterior)
	return posteriors


def get_rational_posterior_matrix_given_one_datum(joint_prob_matrix, prob_true_R, prob_true_not_R, X):
	"""
	Get the posterior matrix given a fully rational updating of beliefs
	"""
	
	joint_prob_d_given_h = get_joint_prob_matrix_d_given_h(prob_true_R, prob_true_not_R, X)
	joint_prob_matrix = get_joint_posterior(joint_prob_matrix, joint_prob_d_given_h)
	return joint_prob_matrix

def get_indy_posterior_matrix_given_one_datum(joint_prob_matrix, prob_true_R, prob_true_not_R, X):
	"""
	Get the posterior matrix given an updating of beliefs with an independence approximation
	"""
	# Independence approximation applied
	prob_H = joint_prob_matrix[0][0] + joint_prob_matrix[1][0]
	prob_R = joint_prob_matrix[0][0] + joint_prob_matrix[0][1]

	independent_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)	

	joint_prob_d_given_h = get_joint_prob_matrix_d_given_h(prob_true_R, prob_true_not_R, X)
	joint_prob_matrix = get_joint_posterior(independent_prob_matrix, joint_prob_d_given_h)

	return joint_prob_matrix

def get_simple_posterior_matrix_given_one_datum(joint_prob_matrix, prob_true_R, prob_true_not_R, X):
	"""
	Get the posterior matrix given a simple updating of beliefs, not updating source reliability
	This looks complciated but its actually simpler than the otehr models.
	We are doing all the steps in this one function 
	"""
	# Extract H and R
	# prob_H is a vector with the prob_h and prob_not H
	prob_H = np.sum(joint_prob_matrix, axis=0)
	prob_H_norm = prob_H/np.sum(prob_H)

	# prob_R will stay the same, jsut grab it from the joint probability matrix
	prob_R = np.sum(joint_prob_matrix, axis=1)
	prob_R_norm = prob_R/np.sum(prob_R)

	# Prob D given H only
	if X == 1:
		joint_prob_d_given_h = np.array([(prob_true_not_R+prob_true_R)*prob_R_norm[0], 1-(prob_true_not_R+prob_true_R)*prob_R_norm[1]])
	if X == 0:
		joint_prob_d_given_h = np.array([1-(prob_true_not_R+prob_true_R)*prob_R_norm[1], (prob_true_not_R+prob_true_R)*prob_R_norm[0]])

	prob_d_given_h = prob_H_norm * joint_prob_d_given_h
	prob_d_given_h_norm = prob_d_given_h/np.sum(prob_d_given_h)

	# Convert back to joint matrix form, to match the foramt of the other models
	joint_prob_matrix = get_joint_prior_matrix(prob_R_norm[0], prob_d_given_h_norm[0])
	
	return joint_prob_matrix


def plot_biased_assimilation_and_evaluation(prob_H, prob_R, prob_true_R, prob_true_not_R):

	fig = plt.figure()

	M = [1,1,1,1,1]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	plt.title(M)
	ax1 = plt.subplot(221)
	# Top left - source reliability give confirmatory evidence

	plt.title("Confirmatory Evidence")

	plt.ylabel("P(R|D)")
	index_pos_r = 0

	simple_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	plt.legend(prop={'size': 7})

	ax2 = plt.subplot(223)

	index_pos_h = 0

	plt.ylabel("P(H|D)")
	plt.xlabel("Observations, $D=[1,1,1,1,1]$")

	simple_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_h, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_h, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_h, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_h, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)

	plt.legend(prop={'size': 7})

	M = [0,0,0,0,0]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	ax3 = plt.subplot(222)

	plt.title("Disconfirmatory Evidence")

	plt.ylabel("P(R|D)")
	index_pos_r = 0

	simple_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	plt.legend(prop={'size': 7})

	ax4 = plt.subplot(224)

	index_pos_h = 0

	plt.ylabel("P(H|D)")
	plt.xlabel("Observations, $D=[0,0,0,0,0]$")

	simple_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_h, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_h, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_h, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_h, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)

	plt.legend(prop={'size': 7})

	axs = [ax1, ax2, ax3, ax4]
	ax_labels = ["a", "c", "b", "d"]
	for ax_index in range(4):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		ax.text(0.0, 1.0, ax_label, transform=ax.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")

	plt.tight_layout()
	plt.savefig("images/biased_evaluation_and_assimilation.png")
	plt.show()


def plot_belief_perseverance(prob_H, prob_R, prob_true_R, prob_true_not_R, different_source=True):
	"""
	Some confirmatory data first, followed by disconfirmatory data from a different source
	"""

	M1 = [1,1,1,1,1]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M1:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		


	M2 = [0,0,0,0,0]

	if different_source:
		# New message source, with a new prior reliability of P(R)=0.5.
		# 2nd message source is independent of 1st mesage source, so no problem to mkae marginals of H at this point 
		rational_prob_H = rational_joint_prob_matrix[0][0] + rational_joint_prob_matrix[1][0]
		rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, rational_prob_H)
		
		indy_prob_H = indy_joint_prob_matrix[0][0] + indy_joint_prob_matrix[1][0]
		indy_joint_prob_matrix  = get_joint_prior_matrix(prob_R, indy_prob_H)

		simple_prob_H = simple_joint_prob_matrix[0][0] + simple_joint_prob_matrix[1][0]
		simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, simple_prob_H)
	

	for X in M2:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	plt.ylabel("P(H|D)")
	if different_source:
		plt.xlabel("Observations, $D={}, D'={}$".format(M1, M2))
	else:
		plt.xlabel("Observations, $D=[1,1,1,1,1,0,0,0,0,0]$")

	index_pos_h = 0

	simple_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), simple_probs_h, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M1) + len(M2) + 1), simple_probs_h, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), rational_probs_h, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M1) + len(M2) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), indy_probs_h, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M1) + len(M2) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)

	plt.grid(axis="y")
	plt.legend()
	plt.tight_layout()
	plt.savefig("images/belief_perseverance_different_source_{}.png".format(different_source))
	plt.show()


def plot_attitude_polarisation(prob_H, prob_R, prob_true_R, prob_true_not_R, different_source=True):
	"""
	Two starting priors - strong for and against. 
	See data from two differnet sources
	"""

	M1 = [1,1,1,1,1]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M1:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		


	M2 = [0,0,0,0,0]

	if different_source:
		# New message source, with a new prior reliability of P(R)=0.5.
		# 2nd message source is independent of 1st mesage source, so no problem to mkae marginals of H at this point 
		rational_prob_H = rational_joint_prob_matrix[0][0] + rational_joint_prob_matrix[1][0]
		rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, rational_prob_H)
		
		indy_prob_H = indy_joint_prob_matrix[0][0] + indy_joint_prob_matrix[1][0]
		indy_joint_prob_matrix  = get_joint_prior_matrix(prob_R, indy_prob_H)

		simple_prob_H = simple_joint_prob_matrix[0][0] + simple_joint_prob_matrix[1][0]
		simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, simple_prob_H)
	

	for X in M2:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	plt.ylabel("P(H|D)")

	index_pos_h = 0

	simple_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), simple_probs_h, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M1) + len(M2) + 1), simple_probs_h, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), rational_probs_h, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M1) + len(M2) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), indy_probs_h, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M1) + len(M2) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)	

	# And for those who are initially disbelieving in HL

	prob_H = 1-prob_H

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]


	M1 = [1,1,1,1,1]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M1:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		


	M2 = [0,0,0,0,0]

	if different_source:
		# New message source, with a new prior reliability of P(R)=0.5.
		# 2nd message source is independent of 1st mesage source, so no problem to mkae marginals of H at this point 
		rational_prob_H = rational_joint_prob_matrix[0][0] + rational_joint_prob_matrix[1][0]
		rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, rational_prob_H)
		
		indy_prob_H = indy_joint_prob_matrix[0][0] + indy_joint_prob_matrix[1][0]
		indy_joint_prob_matrix  = get_joint_prior_matrix(prob_R, indy_prob_H)

		simple_prob_H = simple_joint_prob_matrix[0][0] + simple_joint_prob_matrix[1][0]
		simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, simple_prob_H)
	

	for X in M2:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	plt.ylabel("P(H|D)")
	if different_source:
		plt.xlabel("Observations, $D={}$, $D'={}$".format(M1, M2))

	index_pos_h = 0

	simple_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), simple_probs_h, linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M1) + len(M2) + 1), simple_probs_h, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), rational_probs_h, linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M1) + len(M2) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M1) + len(M2) + 1), indy_probs_h, linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M1) + len(M2) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)

	plt.yticks(np.arange(0, 1.1, 0.1))

	plt.grid(axis="y")

	plt.legend()
	plt.tight_layout()
	plt.savefig("images/attitude_polarisation.png")

	plt.show()


def get_likelihood_ratio_H(prob_true_R, prob_true_not_R, joint_prob_matrix, D="1"):
	"""
	See SI of the paper for a derivation
	Likelihood rations need to be calcualted interms of P(H,R)
	"""
	if D == 1:
		likelihoods = get_joint_prob_matrix_d_given_h(prob_true_R = prob_true_R, prob_true_not_R = prob_true_not_R, X=D)
		prob_d_given_h = (likelihoods[0][0]*joint_prob_matrix[0][0] + likelihoods[1][0]*joint_prob_matrix[1][0])/(joint_prob_matrix[0][0] + joint_prob_matrix[1][0])
		prob_d_given_not_h = (likelihoods[0][1]*joint_prob_matrix[0][1] + likelihoods[1][1]*joint_prob_matrix[1][1])/(joint_prob_matrix[0][1] + joint_prob_matrix[1][1])
		likelihood_ratio = prob_d_given_h/prob_d_given_not_h
		return likelihood_ratio

	if D == 0:
		likelihoods = get_joint_prob_matrix_d_given_h(prob_true_R = prob_true_R, prob_true_not_R = prob_true_not_R, X=D)
		prob_d_given_h = (likelihoods[0][0]*joint_prob_matrix[0][0] + likelihoods[1][0]*joint_prob_matrix[1][0])/(joint_prob_matrix[0][0] + joint_prob_matrix[1][0])
		prob_d_given_not_h = (likelihoods[0][1]*joint_prob_matrix[0][1] + likelihoods[1][1]*joint_prob_matrix[1][1])/(joint_prob_matrix[0][1] + joint_prob_matrix[1][1])
		likelihood_ratio = prob_d_given_h/prob_d_given_not_h
		return likelihood_ratio


def get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_prob_matrix):
	"""
	See SI for derivation
	"""

	diagnosticity_of_question = 0

	for D in [0, 1]:

		likelihoods = get_joint_prob_matrix_d_given_h(prob_true_R = prob_true_R, prob_true_not_R = prob_true_not_R, X=D)
		unnormed_posterior = np.multiply(joint_prob_matrix, likelihoods)
		prob_d_1 = np.sum(unnormed_posterior)

		likelihood_ratio = get_likelihood_ratio_H(prob_true_R, prob_true_not_R, joint_prob_matrix, D=D)

		diagnosticity_of_question += prob_d_1 * np.abs(np.log(likelihood_ratio))
	return diagnosticity_of_question


def plot_selection_sources(prob_H, prob_R, prob_true_R, prob_true_not_R,):
	"""
	Compute diagnosticity of two sources, one for and one against
	"""

	fig = plt.figure()


	M1 = [1,1,1,1,1]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M1:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	rational_diagnosticity_history_confirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in rational_probs_history]
	indy_diagnosticity_history_confirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in indy_probs_history]
	simple_diagnosticity_history_confirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in simple_probs_history]


	M2 = [0,0,0,0,0]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M2:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		


	rational_diagnosticity_history_disconfirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in rational_probs_history]
	indy_diagnosticity_history_disconfirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in indy_probs_history]
	simple_diagnosticity_history_disconfirmatory = [get_diagnosticity_of_question(prob_true_R, prob_true_not_R, joint_probs) for joint_probs in simple_probs_history]


	ax1 = plt.subplot(231)

	plt.plot(range(6), simple_diagnosticity_history_confirmatory, label=r"$D_{for}$", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), simple_diagnosticity_history_disconfirmatory, label=r"$D_{against}$", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), simple_diagnosticity_history_confirmatory, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)
	plt.scatter(range(6), simple_diagnosticity_history_disconfirmatory, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)


	plt.ylabel("Source Diagnosticity")

	plt.xlabel("Observations")

	plt.title("Simple")

	plt.legend(prop={'size': 8})


	ax2 = plt.subplot(232, sharey=ax1)

	plt.plot(range(6), rational_diagnosticity_history_confirmatory, label=r"$D_{for}$", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), rational_diagnosticity_history_disconfirmatory, label=r"$D_{against}$", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_AGAINST)
	plt.scatter(range(6), rational_diagnosticity_history_confirmatory, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)
	plt.scatter(range(6), rational_diagnosticity_history_disconfirmatory, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)



	plt.xlabel("Observations")

	plt.legend(prop={'size': 8})

	plt.title("Rational")

	ax3 = plt.subplot(233, sharey=ax1)

	plt.plot(range(6), indy_diagnosticity_history_confirmatory, label=r"$D_{for}$", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), indy_diagnosticity_history_disconfirmatory, label=r"$D_{against}$", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), indy_diagnosticity_history_confirmatory, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)
	plt.scatter(range(6), indy_diagnosticity_history_disconfirmatory, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)


	plt.xlabel("Observations")

	plt.title("BIASR")

	plt.legend(prop={'size': 8})


	ax4 = plt.subplot(212)

	indy_ratio = np.array(indy_diagnosticity_history_confirmatory)/np.array(indy_diagnosticity_history_disconfirmatory)
	rational_ratio = np.array(rational_diagnosticity_history_confirmatory)/np.array(rational_diagnosticity_history_disconfirmatory)
	simple_ratio = np.array(simple_diagnosticity_history_confirmatory)/np.array(simple_diagnosticity_history_disconfirmatory)

	plt.plot(range(6), simple_ratio, label=r"Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=2)
	plt.scatter(range(6), simple_ratio, marker=MARKER_SIMPLE, color=COLOR_SIMPLE)

	plt.plot(range(6), rational_ratio, label=r"Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=2)
	plt.scatter(range(6), rational_ratio, marker=MARKER_RATIONAL, color=COLOR_RATIONAL)


	plt.plot(range(6), indy_ratio, label=r"BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=2)
	plt.scatter(range(6), indy_ratio, marker=MARKER_INDY, color=COLOR_INDY)

	plt.legend()

	
	plt.xlabel("Observations, $D_{for}=[1,1,1,1,1]$, $D_{against}=[0,0,0,0,0]$")
	plt.ylabel("Diagnosticity Ratio (for/against)")

	axs = [ax1, ax2, ax3, ax4]
	ax_labels = ["a", "b", "c", "d"]
	for ax_index in range(4):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		ax.text(0.0, 1.0, ax_label, transform=ax.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")

	plt.tight_layout()

	plt.savefig("images/confirmation_bias_selection_sources.png")

	plt.show()


def plot_biased_assimilation_and_evaluation_joint_distribution(prob_H, prob_R, prob_true_R, prob_true_not_R, M=[0,0,0,0,0]):

	fig_width, fig_height = plt.gcf().get_size_inches()

	fig = plt.figure(figsize=(fig_width*1.5, fig_height*1.5), constrained_layout=True)

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	ax1 = plt.subplot(331)

	plt.ylabel("$P(H=1, R=1|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 0
	index_pos_r = 0

	simple_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [prob_matrix[index_pos_h, index_pos_r]for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	ax2 = plt.subplot(332)

	plt.ylabel("$P(H=1, R=0|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 1
	index_pos_r = 0

	simple_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [prob_matrix[index_pos_h, index_pos_r]for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	ax4 = plt.subplot(334)

	plt.ylabel("$P(H=0, R=1|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 0
	index_pos_r = 1

	simple_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [prob_matrix[index_pos_h, index_pos_r]for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)


	ax5 = plt.subplot(335)

	plt.ylabel("$P(H=0, R=0|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 1
	index_pos_r = 1

	simple_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [prob_matrix[index_pos_h, index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [prob_matrix[index_pos_h, index_pos_r]for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)


	ax3 = plt.subplot(333)

	plt.ylabel("$P(H=1|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 0

	simple_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)


	ax6 = plt.subplot(336)

	plt.ylabel("$P(H=0|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_h = 1

	simple_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)


	ax7 = plt.subplot(337)

	plt.ylabel("$P(R=1|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_r = 0

	simple_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	ax8 = plt.subplot(338)

	plt.ylabel("$P(R=0|D)$")
	plt.xlabel("Observations, D={}".format(M))
	index_pos_r = 1

	simple_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in simple_probs_history]
	plt.plot(range(len(M) + 1), simple_probs_r, label="Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE)
	plt.scatter(range(len(M) + 1), simple_probs_r, color=COLOR_SIMPLE, marker=MARKER_SIMPLE)

	rational_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_r, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_r, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	indy_probs_r = [np.sum(prob_matrix, axis=1)[index_pos_r] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_r, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_r, color=COLOR_INDY, marker=MARKER_INDY)

	# Shrink current axis by 20%
	box = ax8.get_position()

	# Put a legend to the right of the current axis
	
	axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
	ax_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
	for ax_index in range(8):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		ax.text(0.0, 1.0, ax_label, transform=ax.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")


	plt.tight_layout()
	plt.legend(bbox_to_anchor=(1.3,1), loc="upper left")
	plt.savefig("images/joint_probs_updating_{}.png".format(M).replace(" ", "_").replace(",",""))

	plt.show()


def plot_all_simulations():

	prob_H = 0.8
	prob_R = 0.5
	prob_true_R = 0.75
	prob_true_not_R = 0.5

	plot_biased_assimilation_and_evaluation_joint_distribution(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R, M=[1,1,1,1,1])
	plot_biased_assimilation_and_evaluation_joint_distribution(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R, M=[0,0,0,0,0])

	plot_attitude_polarisation(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R)
	plot_biased_assimilation_and_evaluation(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R)	
	plot_biased_assimilation_and_evaluation(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R)
	plot_selection_sources(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R)

	prob_H = 0.5
	
	plot_belief_perseverance(prob_H=prob_H, prob_R=prob_R, prob_true_R=prob_true_R, prob_true_not_R=prob_true_not_R, different_source=True)


def plot_redlawsk_replication():

	prob_H = 0.8
	prob_R = 0.5
	prob_true_R = 0.75
	prob_true_not_R = 0.35

	fig = plt.figure()

	ax1 = plt.subplot(121)

	y1 = [78, 83]

	plt.plot(range(2), y1, label="Online", color=COLOR_INDY)
	plt.scatter(range(2), y1,  color=COLOR_INDY)


	y2 = [78, 71]

	plt.plot(range(2), y2, label="Memory", color=COLOR_RATIONAL, linestyle=LINESTYLE_RATIONAL)	
	plt.scatter(range(2), y2, color=COLOR_RATIONAL, linestyle=LINESTYLE_RATIONAL)

	plt.ylabel("Estimated Means of Ratings for Candidates")
	plt.xlabel("No Incongruent Info         Incongruent Info")
	plt.xticks([])
	
	plt.legend()

	M = [0,0,0,0,0]

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	simple_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history = [rational_joint_prob_matrix]
	indy_probs_history = [indy_joint_prob_matrix]
	simple_probs_history = [simple_joint_prob_matrix]

	for X in M:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		rational_probs_history.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		indy_probs_history.append(indy_joint_prob_matrix)		

		simple_joint_prob_matrix = get_simple_posterior_matrix_given_one_datum(simple_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
		simple_probs_history.append(simple_joint_prob_matrix)		

	ax2 = plt.subplot(122)

	index_pos_h = 0

	plt.ylabel("P(H|D)")
	plt.xlabel("Observations, $D=[0,0,0,0,0]$")


	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history]
	plt.plot(range(len(M) + 1), indy_probs_h, label="BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)


	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history]
	plt.plot(range(len(M) + 1), rational_probs_h, label="Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)



	axs = [ax1, ax2]
	ax_labels = ["a", "b"]
	for ax_index in range(2):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		ax.text(0.0, 1.0, ax_label, transform=ax.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")

	plt.xticks([])

	plt.legend()

	plt.tight_layout()
	plt.savefig("images/redlawsk_replication.png")
	plt.show()


def get_joint_prob_matrix_d_given_h_carlson(X):
	"""
	For the carlson replication
	"""

	prob_D_given_H_R = [0.4, 0.4, 0.2]
	prob_D_given_not_H_R = [0.5, 0.4, 0.1]
	prob_D_given_H_not_R = [0.4, 0.5, 0.1]
	prob_D_given_not_H_not_R = [0.45, 0.45, 0.1]

	joint_prob_d_given_h = np.array([
		[prob_D_given_H_R[X], prob_D_given_not_H_R[X]],
		[prob_D_given_H_not_R[X], prob_D_given_not_H_not_R[X]]
		])

	return joint_prob_d_given_h


def get_rational_posterior_matrix_given_one_datum_carlson(joint_prob_matrix, X):
	"""
	Get the posterior matrix given a fully rational updating of beliefs
	For the Carlson replication only
	"""
	
	joint_prob_d_given_h = get_joint_prob_matrix_d_given_h_carlson(X)
	joint_prob_matrix = get_joint_posterior(joint_prob_matrix, joint_prob_d_given_h)
	return joint_prob_matrix

def get_indy_posterior_matrix_given_one_datum_carlson(joint_prob_matrix, X):
	"""
	Get the posterior matrix given an updating of beliefs with an independence approximation
	"""
	# Independence approximation applied
	prob_H = joint_prob_matrix[0][0] + joint_prob_matrix[1][0]
	prob_R = joint_prob_matrix[0][0] + joint_prob_matrix[0][1]

	independent_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)	

	joint_prob_d_given_h = get_joint_prob_matrix_d_given_h_carlson(X)
	joint_prob_matrix = get_joint_posterior(independent_prob_matrix, joint_prob_d_given_h)

	return joint_prob_matrix


def plot_carlson_replication():

	"""
	D can be 0,1 or 2. 
	The P(D|H,R) is given in the function get_joint_prob_matrix_d_given_h_carlson
	You probabbly wan tto report taht as well in the figure. 
	"""

	fig_width, fig_height = plt.gcf().get_size_inches()

	fig = plt.figure(figsize=(fig_width*2, fig_height), constrained_layout=True)
	

	prob_H = 0.5
	prob_R = 0.5

	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history_M1 = [rational_joint_prob_matrix]
	indy_probs_history_M1 = [indy_joint_prob_matrix]

	M1 = [2,1,1,1,1,1]

	for X in M1:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum_carlson(rational_joint_prob_matrix, X)
		rational_probs_history_M1.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum_carlson(indy_joint_prob_matrix, X)
		indy_probs_history_M1.append(indy_joint_prob_matrix)		


	rational_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)
	indy_joint_prob_matrix = get_joint_prior_matrix(prob_R, prob_H)

	rational_probs_history_M2 = [rational_joint_prob_matrix]
	indy_probs_history_M2 = [indy_joint_prob_matrix]

	M2 = [1,1,1,2,1,1]

	for X in M2:
		
		rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum_carlson(rational_joint_prob_matrix, X)
		rational_probs_history_M2.append(rational_joint_prob_matrix)

		indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum_carlson(indy_joint_prob_matrix, X)
		indy_probs_history_M2.append(indy_joint_prob_matrix)		

	index_pos_h = 0

	# Indy condition
	ax1 = plt.subplot(131)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history_M1]
	plt.plot(range(len(M1) + 1), indy_probs_h, label=r"BIASR $M_1$", linestyle=LINESTYLE_INDY, color=COLOR_INDY)
	plt.scatter(range(len(M1) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_INDY)

	indy_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in indy_probs_history_M2]
	plt.plot(range(len(M1) + 1), indy_probs_h, label=r"BIASR $M_2$", linestyle=LINESTYLE_RATIONAL, color=COLOR_INDY)
	plt.scatter(range(len(M1) + 1), indy_probs_h, color=COLOR_INDY, marker=MARKER_RATIONAL)

	plt.ylabel("P(H = Good Restaurant)")

	plt.xlabel("M")

	plt.legend()

	# Rational condition
	ax2 = plt.subplot(132, sharey=ax1)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history_M1]
	plt.plot(range(len(M2) + 1), rational_probs_h, label=r"Rational $M_1$", linestyle=LINESTYLE_INDY, color=COLOR_RATIONAL)
	plt.scatter(range(len(M2) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_INDY)

	rational_probs_h = [np.sum(prob_matrix, axis=0)[index_pos_h] for prob_matrix in rational_probs_history_M2]
	plt.plot(range(len(M2) + 1), rational_probs_h, label=r"Rational $M_2$", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL)
	plt.scatter(range(len(M2) + 1), rational_probs_h, color=COLOR_RATIONAL, marker=MARKER_RATIONAL)

	plt.xlabel("M")

	

	plt.legend()


	ax3 = plt.subplot(133)

	draw_table_carlson(ax3)


	axs = [ax1, ax2, ax3]
	ax_labels = ["a", "b"]
	for ax_index in range(2):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		ax.text(0.0, 1.0, ax_label, transform=ax.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")

	ax3.text(0.0, 0.7, "c", transform=ax3.transAxes + trans,
				fontsize='large', va='bottom', weight="bold")


	ax3.text(0.3, 0.15, r"$M_1 = [2,1,1,1,1,1]$", transform=ax3.transAxes + trans,
				fontsize='large', va='bottom')

	ax3.text(0.3, 0.05, r"$M_2 = [1,1,1,2,1,1]$", transform=ax3.transAxes + trans,
				fontsize='large', va='bottom')


	plt.savefig("images/carlson_replication.png", dpi=300)

	plt.show()


def draw_table_carlson(ax):

	table_data = [
		[1,1,0.4, 0.4, 0.2],
		[0,1,0.5, 0.4, 0.1],
		[1,0,0.4, 0.5, 0.1],
		[0,0,0.45, 0.45, 0.1],
		]

	collabel=("H", "R", "P(D=0)", "P(D=1)", "P(D=2)")
	ax.axis('tight')
	ax.axis('off')
	the_table = ax.table(cellText=table_data,
		colLabels=collabel,
		cellLoc="center",
		colLoc="center",
		edges="closed",
		loc='center')

	the_table.set_fontsize(32)
	the_table.scale(1.2, 1.5)


if __name__=="__main__":
	plot_carlson_replication()