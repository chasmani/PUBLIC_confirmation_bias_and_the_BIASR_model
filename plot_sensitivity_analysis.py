

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list('', ['#ff3d3d', 'white', '#74ff52'])

from plot_model_simulations import get_joint_prior_matrix, get_rational_posterior_matrix_given_one_datum, get_indy_posterior_matrix_given_one_datum



def plot_biased_evaluation_robustness():


	number_hs = 201
	number_rs = 201

	prob_true_R = 0.75
	prob_true_not_R = 0.5

	Hs = np.linspace(0,1, num=number_hs)
	Rs = np.linspace(0,1,num=number_rs)

	bias_posts = np.empty((number_hs, number_rs))
	
	D1 = [1,1]
	D2 = [0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]

			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			for X in D1:
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
			indy_prob_r_1 = np.sum(indy_joint_prob_matrix, axis=1)[0]
		
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			for X in D2:
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
			indy_prob_r_2 = np.sum(indy_joint_prob_matrix, axis=1)[0]

			bias_posts[h_ind][r_ind] = indy_prob_r_1 - indy_prob_r_2

	plt.subplot(1,2,1)

	bias_posts = np.round(bias_posts, 5)

	df = pd.DataFrame(bias_posts, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(np.round(ax.get_xticks()[::2],2))
	ax.set_yticks(np.round(ax.get_yticks()[::5],2))
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")


	plt.gca().invert_yaxis()

	plt.title("D=[1,1], D'=[0,0]\n")


	bias_posts = np.empty((number_hs, number_rs))
	
	D1 = [1,1,1,1,1]
	D2 = [0,0,0,0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]

			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			for X in D1:
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
			indy_prob_r_1 = np.sum(indy_joint_prob_matrix, axis=1)[0]
		

			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			for X in D2:
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
			indy_prob_r_2 = np.sum(indy_joint_prob_matrix, axis=1)[0]	

			bias_posts[h_ind][r_ind] = indy_prob_r_1 - indy_prob_r_2

	plt.subplot(1,2,2)

	bias_posts = np.round(bias_posts, 5)

	df = pd.DataFrame(bias_posts, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(np.round(ax.get_xticks()[::2],2))
	ax.set_yticks(np.round(ax.get_yticks()[::5],2))
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")


	plt.gca().invert_yaxis()

	plt.title("D=[1,1,1,1,1], D'=[0,0,0,0,0]\n")


	plt.tight_layout()

	plt.savefig("images/sensitivity_biased_evaluation.png", dpi=600)



	plt.show()





def plot_belief_perseverance_robustness():

	number_hs = 201
	number_rs = 201

	prob_true_R = 0.75
	prob_true_not_R = 0.5

	Hs = np.linspace(0,1, num=number_hs)
	Rs = np.linspace(0,1,num=number_rs)

	joint_posts = np.empty((number_hs, number_rs))
	indy_posts = np.empty((number_hs, number_rs))

	D1 = [1,1]
	D2 = [0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]
			print(prior_H, prior_R)

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in D1:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			print(rational_joint_prob_matrix)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			
			# Set belief distirbtuion with the new source
			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, rational_prob_h)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, indy_prob_h)

			for X in D2:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			joint_posts[h_ind][r_ind] = rational_prob_h
			indy_posts[h_ind][r_ind] = indy_prob_h

	plt.subplot(1,2,1)

	print(joint_posts)

	absolute_bias = indy_posts - joint_posts
	#print(absolute_bias)

	df = pd.DataFrame(absolute_bias, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::5])
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")


	plt.gca().invert_yaxis()

	plt.title("D=[1,1], D'=[0,0]\n")

	joint_posts = np.empty((number_hs, number_rs))
	indy_posts = np.empty((number_hs, number_rs))

	D1 = [1,1,1,1,1]
	D2 = [0,0,0,0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]
			print(prior_H, prior_R)

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in D1:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			print(rational_joint_prob_matrix)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			
			# Set belief distirbtuion with the new source
			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, rational_prob_h)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, indy_prob_h)

			for X in D2:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			joint_posts[h_ind][r_ind] = rational_prob_h
			indy_posts[h_ind][r_ind] = indy_prob_h

	plt.subplot(1,2,2)

	print(joint_posts)

	absolute_bias = indy_posts - joint_posts
	#print(absolute_bias)

	df = pd.DataFrame(absolute_bias, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::5])
	

	plt.gca().invert_yaxis()

	plt.title("D=[1,1,1,1,1], D'=[0,0,0,0,0]\n")

	plt.tight_layout()

	plt.savefig("images/sensitivity_belief_perseverance.png", dpi=600)

	plt.show()


def plot_attitude_polarisation_robustness():

	number_hs = 201
	number_rs = 201

	prob_true_R = 0.75
	prob_true_not_R = 0.5

	Hs = np.linspace(0,1, num=number_hs)
	Rs = np.linspace(0,1,num=number_rs)

	bias_posts = np.empty((number_hs, number_rs))
	
	D1 = [1,1]
	D2 = [0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]
			print(prior_H, prior_R)

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in D1:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			print(rational_joint_prob_matrix)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			
			# Set belief distirbtuion with the new source
			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, rational_prob_h)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, indy_prob_h)

			for X in D2:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]

			bias_posts[h_ind][r_ind] = indy_prob_h - prior_H


	plt.subplot(1,2,1)


	df = pd.DataFrame(bias_posts, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::5])
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")


	plt.gca().invert_yaxis()

	plt.title("D=[1,1], D'=[0,0]\n")

	bias_posts = np.empty((number_hs, number_rs))
	
	D1 = [1,1,1,1,1]
	D2 = [0,0,0,0,0]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]
			print(prior_H, prior_R)

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in D1:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			print(rational_joint_prob_matrix)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			
			# Set belief distirbtuion with the new source
			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, rational_prob_h)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, indy_prob_h)

			for X in D2:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			bias_posts[h_ind][r_ind] = indy_prob_h - prior_H

	
	plt.subplot(1,2,2)


	df = pd.DataFrame(bias_posts, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::5])
	

	plt.gca().invert_yaxis()

	plt.title("D=[1,1,1,1,1], D'=[0,0,0,0,0]\n")

	plt.tight_layout()

	plt.savefig("images/sensitivity_attitude_polarisation.png", dpi=600)
	
	plt.show()




def plot_biased_assimilation_robustness():

	number_hs = 201
	number_rs = 201

	prob_true_R = 0.75
	prob_true_not_R = 0.5

	Hs = np.linspace(0,1, num=number_hs)
	Rs = np.linspace(0,1,num=number_rs)

	joint_posts_2_datums = np.empty((number_hs, number_rs))
	indy_posts_2_datums = np.empty((number_hs, number_rs))

	joint_posts_5_datums = np.empty((number_hs, number_rs))
	indy_posts_5_datums = np.empty((number_hs, number_rs))

	M1 = [1,1]
	M2 = [1,1,1]

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in M1:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]

			joint_posts_2_datums[h_ind][r_ind] = rational_prob_h
			indy_posts_2_datums[h_ind][r_ind] = indy_prob_h


			for X in M2:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]

			joint_posts_5_datums[h_ind][r_ind] = rational_prob_h
			indy_posts_5_datums[h_ind][r_ind] = indy_prob_h

	ax1 = plt.subplot(121)

	absolute_bias_2_datums = indy_posts_2_datums - joint_posts_2_datums

	df = pd.DataFrame(absolute_bias_2_datums, index=Hs, columns=Rs)

	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)


	ax = plt.gca()
	n = 5  # Set your desired interval

	print(ax.get_xticks()[::2])

	ax.axhline(y=100.5, color="gray", linestyle="--")

	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::n])



	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")


	plt.gca().invert_yaxis()

	plt.title("D=[1,1]")

	ax2 = plt.subplot(122)

	absolute_bias_5_datums = indy_posts_5_datums - joint_posts_5_datums
	df = pd.DataFrame(absolute_bias_5_datums, index=Hs, columns=Rs)
	
	sns.heatmap(df, norm=mcolors.TwoSlopeNorm(0), cmap=cmap)

	
	plt.xlabel("prior P(R)")
	plt.ylabel("prior P(H)")

	ax = plt.gca()
	ax.set_xticks(ax.get_xticks()[::2])
	ax.set_yticks(ax.get_yticks()[::n])

	ax.axhline(y=100.5, color="gray", linestyle="--")

	plt.gca().invert_yaxis()

	plt.title("D=[1,1,1,1,1]")

	plt.tight_layout()

	plt.savefig("images/sensitivity_biased_assimilation.png", dpi=600)

	plt.show()


def plot_biased_assimilation_robustness_posterior(M=[1,1]):

	number_hs = 51
	number_rs = 51

	prob_true_R = 0.75
	prob_true_not_R = 0.5

	Hs = np.linspace(0.01,0.99, num=number_hs)
	Rs = np.linspace(0.01,0.99, num=number_rs)

	joint_posts = np.empty((number_hs, number_rs))
	indy_posts = np.empty((number_hs, number_rs))

	for h_ind in range(number_hs):
		for r_ind in range(number_rs):
			prior_H = Hs[h_ind]
			prior_R = Rs[r_ind]
			print(prior_H, prior_R)

			rational_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)
			indy_joint_prob_matrix = get_joint_prior_matrix(prior_R, prior_H)

			for X in M:
				rational_joint_prob_matrix = get_rational_posterior_matrix_given_one_datum(rational_joint_prob_matrix, prob_true_R, prob_true_not_R, X)
				indy_joint_prob_matrix = get_indy_posterior_matrix_given_one_datum(indy_joint_prob_matrix, prob_true_R, prob_true_not_R, X)

			print(rational_joint_prob_matrix)

			rational_prob_r = np.sum(rational_joint_prob_matrix, axis=1)[0]
			indy_prob_r = np.sum(indy_joint_prob_matrix, axis=1)[0]
			bias_r = indy_prob_r - rational_prob_r

			rational_prob_h = np.sum(rational_joint_prob_matrix, axis=0)[0]
			indy_prob_h = np.sum(indy_joint_prob_matrix, axis=0)[0]
			bias_h = indy_prob_h - rational_prob_h
			bias_r = indy_prob_r - rational_prob_r

			if bias_h > 0:
				color = "#74ff52"
			else:
				color = "#ff3d3d"
			size = round(np.abs(bias_h)*200) + 1

			plt.scatter(x=indy_prob_r, y=indy_prob_h, color=color, s=size)
	plt.xlabel("Posterior P(R|D)")
	plt.ylabel("Posterior P(H|D)")

	from matplotlib.lines import Line2D
	legend_elements = [Line2D([0], [0], marker='o', color="w", markerfacecolor='#74ff52', label='Positive Bias'),
		Line2D([0], [0], marker='o', color="w", markerfacecolor="#ff3d3d", label='Negative Bias')]

	plt.legend(handles=legend_elements)

	plt.axhline(y=0.5, color="gray", linestyle="--")

	plt.savefig("images/sensitivity_biased_assimilation_posterior.png", dpi=600)
	plt.show()





if __name__=="__main__":
	#plot_biased_assimilation_robustness()
	#plot_biased_assimilation_robustness_posterior()
	#plot_attitude_polarisation_robustness()
	plot_biased_evaluation_robustness()

