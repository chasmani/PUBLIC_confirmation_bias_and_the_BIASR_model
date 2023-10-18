

import numpy as np

PROB_TRUE_R = 0.75
PROB_TRUE_NOT_R = 0.5

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

def get_prob_D_rational(prior, D, source):
	
	unnormed_posterior = get_unnormed_posterior_rational(prior, D, source)
	return np.sum(unnormed_posterior)

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

def get_prob_D_indy(prior, D, source):
	
	unnormed_posterior = get_unnormed_posterior_indy(prior, D, source)
	return np.sum(unnormed_posterior)

def get_posterior_simple(prior, D, prob_R):

	unnormed_posterior = get_unnormed_posterior_simple(prior, D, prob_R)
	return unnormed_posterior/np.sum(unnormed_posterior)

def get_unnormed_posterior_simple(prior, D, prob_R):
	
	likelihood = get_likelihood_simple(D=D, prior_R = prob_R)
	# Get prob_H and prob_Rs
	return np.multiply(prior, likelihood)

def get_prob_D_simple(prior, D, prob_R):
	
	unnormed_posterior = get_unnormed_posterior_simple(prior, D, prob_R)
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


def get_entropy(joint_prob_dist):

	prob_H = np.sum(joint_prob_dist, axis=(1,2))

	entropy = 0
	for H in [0,1]:
		if prob_H[H] != 0:
			entropy += - prob_H[H] * np.log(prob_H[H])
	return entropy


def get_expected_information_gain(joint_prob_dist, update_type="rational", source="A"):

	current_entropy = get_entropy(joint_prob_dist)

	expected_entropy = 0
	
	prob_R = 0.5

	for D in [0,1]:
		if update_type == "rational":
			prob_D = get_prob_D_rational(joint_prob_dist, D, source)
			posterior_given_D = get_posterior_rational(joint_prob_dist, D, source)
			entropy_given_D = get_entropy(posterior_given_D)
			expected_entropy += prob_D * entropy_given_D
		elif update_type == "indy":
			prob_D = get_prob_D_indy(joint_prob_dist, D, source)
			posterior_given_D = get_posterior_indy(joint_prob_dist, D, source)
			entropy_given_D = get_entropy(posterior_given_D)
			expected_entropy += prob_D * entropy_given_D
		elif update_type == "simple":
			prob_D = get_prob_D_simple(joint_prob_dist, D, prob_R=prob_R)
			posterior_given_D = get_posterior_simple(joint_prob_dist, D=D, prob_R=prob_R)
			entropy_given_D = get_entropy(posterior_given_D)
			expected_entropy += prob_D * entropy_given_D

	return current_entropy - expected_entropy


def plot_selection_sources_with_information_gain(prob_H=0.75, prob_R=0.5):
	"""
	Compute exected information gain of two sources, one for and one against
	"""

	fig = plt.figure()

	M = [1,0] * 5
	sources = ["A", "B"] * 5

	rational_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	indy_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	simple_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)

	for_rational_information_gain = [get_expected_information_gain(rational_joint_prob, update_type="rational", source="A")]
	against_rational_information_gain = [get_expected_information_gain(rational_joint_prob, update_type="rational", source="B")]
	for_indy_information_gain = [get_expected_information_gain(indy_joint_prob, update_type="indy", source="A")]
	against_indy_information_gain = [get_expected_information_gain(indy_joint_prob, update_type="indy", source="B")]
	for_simple_information_gain = [get_expected_information_gain(simple_joint_prob, update_type="simple", source="A")]
	against_simple_information_gain = [get_expected_information_gain(simple_joint_prob, update_type="simple", source="B")]

	for message_i in range(len(M)):

		D = M[message_i]
		source = sources[message_i]
		
		rational_joint_prob = get_posterior_rational(rational_joint_prob, D, source=source)

		indy_joint_prob = get_posterior_indy(indy_joint_prob, D, source=source)
		print(indy_joint_prob)
		print(np.sum(indy_joint_prob, axis=(0,1)))

		simple_joint_prob = get_posterior_simple(simple_joint_prob, D=D, prob_R=prob_R)

		if message_i % 2 == 1:
			for_rational_information_gain.append(get_expected_information_gain(rational_joint_prob, update_type="rational", source="A"))
			against_rational_information_gain.append(get_expected_information_gain(rational_joint_prob, update_type="rational", source="B"))

			for_indy_information_gain.append(get_expected_information_gain(indy_joint_prob, update_type="indy", source="A"))
			against_indy_information_gain.append(get_expected_information_gain(indy_joint_prob, update_type="indy", source="B"))

			for_simple_information_gain.append(get_expected_information_gain(simple_joint_prob, update_type="simple", source="A"))
			against_simple_information_gain.append(get_expected_information_gain(simple_joint_prob, update_type="simple", source="B"))

	ax1 = plt.subplot(231)

	plt.plot(range(6), for_simple_information_gain, label="for", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_simple_information_gain, label="against", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_simple_information_gain, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)
	plt.scatter(range(6), against_simple_information_gain, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)

	plt.ylabel("Expected Information Gain")
	plt.xlabel("Observations")

	plt.title("Simple")

	plt.legend(prop={'size': 8})

	plt.ylim([0, 0.055])

	ax2 = plt.subplot(232, sharey=ax1)

	plt.plot(range(6), for_rational_information_gain, label="for", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_rational_information_gain, label="against", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_rational_information_gain, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)
	plt.scatter(range(6), against_rational_information_gain, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)

	plt.xlabel("Observations")

	plt.title("Rational")

	plt.legend(prop={'size': 8})

	ax3 = plt.subplot(233, sharey=ax1)

	plt.plot(range(6), for_indy_information_gain, label="for", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_indy_information_gain, label="against", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_indy_information_gain, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)
	plt.scatter(range(6), against_indy_information_gain, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)

	plt.xlabel("Observations")

	plt.title("BIASR")

	plt.legend(prop={'size': 8})


	ax4 = plt.subplot(212)

	indy_ratio = np.array(for_indy_information_gain)/np.array(against_indy_information_gain)
	rational_ratio = np.array(for_rational_information_gain)/np.array(against_rational_information_gain)
	simple_ratio = np.array(for_simple_information_gain)/np.array(against_simple_information_gain)

	plt.plot(range(6), simple_ratio, label=r"Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=2)
	plt.scatter(range(6), simple_ratio, marker=MARKER_SIMPLE, color=COLOR_SIMPLE)

	plt.plot(range(6), rational_ratio, label=r"Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=2)
	plt.scatter(range(6), rational_ratio, marker=MARKER_RATIONAL, color=COLOR_RATIONAL)


	plt.plot(range(6), indy_ratio, label=r"BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=2)
	plt.scatter(range(6), indy_ratio, marker=MARKER_INDY, color=COLOR_INDY)

	plt.legend()

	
	plt.xlabel(r"Observations, $(for, against)$, $D = [(1,0), (1,0), (1,0), (1,0), (1,0)]$")
	plt.ylabel("Information Gain Ratio\n(for/against)")

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

	plt.savefig("images/confirmation_bias_selection_sources_with_information_gain.png")

	plt.show()


def plot_selection_sources_with_diagnosticity(prob_H=0.75, prob_R=0.5):
	"""
	Compute exected information gain of two sources, one for and one against
	"""

	fig = plt.figure()

	M = [1,0] * 5
	sources = ["A", "B"] * 5

	rational_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	indy_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)
	simple_joint_prob = get_joint_prior_two_sources(prob_H, prob_R, prob_R)

	for_rational_information_gain = [get_diagnosticity_of_question(rational_joint_prob, update_type="rational", source="A")]
	against_rational_information_gain = [get_diagnosticity_of_question(rational_joint_prob, update_type="rational", source="B")]
	for_indy_information_gain = [get_diagnosticity_of_question(indy_joint_prob, update_type="indy", source="A")]
	against_indy_information_gain = [get_diagnosticity_of_question(indy_joint_prob, update_type="indy", source="B")]
	for_simple_information_gain = [get_diagnosticity_of_question(simple_joint_prob, update_type="simple", source="A")]
	against_simple_information_gain = [get_diagnosticity_of_question(simple_joint_prob, update_type="simple", source="B")]

	for message_i in range(len(M)):

		D = M[message_i]
		source = sources[message_i]
		
		rational_joint_prob = get_posterior_rational(rational_joint_prob, D, source=source)

		indy_joint_prob = get_posterior_indy(indy_joint_prob, D, source=source)

		simple_joint_prob = get_posterior_simple(simple_joint_prob, D=D, prob_R=prob_R)

		if message_i % 2 == 1:
			for_rational_information_gain.append(get_diagnosticity_of_question(rational_joint_prob, update_type="rational", source="A"))
			against_rational_information_gain.append(get_diagnosticity_of_question(rational_joint_prob, update_type="rational", source="B"))

			for_indy_information_gain.append(get_diagnosticity_of_question(indy_joint_prob, update_type="indy", source="A"))
			against_indy_information_gain.append(get_diagnosticity_of_question(indy_joint_prob, update_type="indy", source="B"))

			for_simple_information_gain.append(get_diagnosticity_of_question(simple_joint_prob, update_type="simple", source="A"))
			against_simple_information_gain.append(get_diagnosticity_of_question(simple_joint_prob, update_type="simple", source="B"))

	ax1 = plt.subplot(231)

	print(for_simple_information_gain)

	plt.plot(range(6), for_simple_information_gain, label="for", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_simple_information_gain, label="against", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_simple_information_gain, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)
	plt.scatter(range(6), against_simple_information_gain, marker=MARKER_SIMPLE, color=COLOR_SIMPLE, s=SIZE_SOURCES)

	plt.ylabel("Diagnosticity of Source")
	plt.xlabel("Observations")

	plt.title("Simple")

	plt.legend(prop={'size': 8})

	plt.ylim([0, 1.1])

	ax2 = plt.subplot(232, sharey=ax1)

	plt.plot(range(6), for_rational_information_gain, label="for", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_rational_information_gain, label="against", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_rational_information_gain, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)
	plt.scatter(range(6), against_rational_information_gain, marker=MARKER_RATIONAL, color=COLOR_RATIONAL, s=SIZE_SOURCES)

	plt.xlabel("Observations")

	plt.title("Rational")

	plt.legend(prop={'size': 8})

	ax3 = plt.subplot(233, sharey=ax1)

	plt.plot(range(6), for_indy_information_gain, label="for", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_FOR)
	plt.plot(range(6), against_indy_information_gain, label="against", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=LINEWIDTH_AGAINST)

	plt.scatter(range(6), for_indy_information_gain, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)
	plt.scatter(range(6), against_indy_information_gain, marker=MARKER_INDY, color=COLOR_INDY, s=SIZE_SOURCES)

	plt.xlabel("Observations")

	plt.title("BIASR")

	plt.legend(prop={'size': 8})


	ax4 = plt.subplot(212)

	indy_ratio = np.array(for_indy_information_gain)/np.array(against_indy_information_gain)
	rational_ratio = np.array(for_rational_information_gain)/np.array(against_rational_information_gain)
	simple_ratio = np.array(for_simple_information_gain)/np.array(against_simple_information_gain)

	plt.plot(range(6), simple_ratio, label=r"Simple", linestyle=LINESTYLE_SIMPLE, color=COLOR_SIMPLE, linewidth=2)
	plt.scatter(range(6), simple_ratio, marker=MARKER_SIMPLE, color=COLOR_SIMPLE)

	plt.plot(range(6), rational_ratio, label=r"Rational", linestyle=LINESTYLE_RATIONAL, color=COLOR_RATIONAL, linewidth=2)
	plt.scatter(range(6), rational_ratio, marker=MARKER_RATIONAL, color=COLOR_RATIONAL)


	plt.plot(range(6), indy_ratio, label=r"BIASR", linestyle=LINESTYLE_INDY, color=COLOR_INDY, linewidth=2)
	plt.scatter(range(6), indy_ratio, marker=MARKER_INDY, color=COLOR_INDY)

	plt.legend()

	
	plt.xlabel(r"Observations, $(for, against)$, $D = [(1,0), (1,0), (1,0), (1,0), (1,0)]$")
	plt.ylabel("Diagnosticity Ratio\n(for/against)")

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

	plt.savefig("images/confirmation_bias_selection_sources_with_diagnosticity.png")

	plt.show()


def get_diagnosticity_of_question(joint_prob_dist, update_type="rational", source="A"):

	diagnosticity_of_question = 0

	for D in [0, 1]:

		if update_type == "rational":
			prob_D = get_prob_D_rational(joint_prob_dist, D, source)
			unnormed_posterior = get_unnormed_posterior_rational(joint_prob_dist, D, source)
		elif update_type == "indy":
			prob_D = get_prob_D_indy(joint_prob_dist, D, source)
			unnormed_posterior = get_unnormed_posterior_indy(joint_prob_dist, D, source)

		elif update_type == "simple":
			prob_D = get_prob_D_simple(joint_prob_dist, D, prob_R=0.5)
			unnormed_posterior = get_unnormed_posterior_simple(joint_prob_dist, D, prob_R=0.5)

		numerator_h_1 = np.sum(unnormed_posterior, axis=(1,2))[1]
		denominator_h_1 = np.sum(joint_prob_dist, axis=(1,2))[1]
		numerator_h_0 = np.sum(unnormed_posterior, axis=(1,2))[0]
		denominator_h_0 = np.sum(joint_prob_dist, axis=(1,2))[0]
		likelihood_ratio =  (numerator_h_1/denominator_h_1)/(numerator_h_0/denominator_h_0)
		
		diagnosticity_of_question += prob_D * np.abs(np.log(likelihood_ratio))

	return diagnosticity_of_question







if __name__=="__main__":
	#print(get_joint_prior_two_sources(0.5, 0.2,0.8))
	plot_selection_sources_with_information_gain()
	plot_selection_sources_with_diagnosticity()