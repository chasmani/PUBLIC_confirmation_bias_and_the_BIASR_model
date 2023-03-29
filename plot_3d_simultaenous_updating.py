


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import networkx as nx

from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
style.use('ggplot')

COLOR_INDY = "#2980b9"
COLOR_RATIONAL = "#e67e22"
COLOR_SIMPLE = "#6ab04c"

def multiple_plots():


	# setup the figure and axes
	fig = plt.figure(figsize=(8, 3))

	ax1 = fig.add_subplot(221, projection='3d')


	ax2 = fig.add_subplot(223, projection='3d')
	ax3 = fig.add_subplot(224, projection='3d')

	# fake data
	_x = np.arange(4)
	_y = np.arange(5)
	_xx, _yy = np.meshgrid(_x, _y)
	x, y = _xx.ravel(), _yy.ravel()

	top = x + y
	bottom = np.zeros_like(top)
	width = depth = 1

	ax2.bar3d(x, y, bottom, width, depth, top, shade=True)


	ax3.bar3d(x, y, bottom, width, depth, top, shade=False)

	plt.show()

def one_plot():

	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib import style
	style.use('ggplot')

	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')

	fig.set_facecolor('white')
	ax1.set_facecolor('white') 

	x3 = [0,0,0,2,2,2,4,4]
	y3 = [5,3,2,5,3,2,5,3]
	z3 = np.zeros(8)

	dx = [1,1,1,1,1,1,0.1,0.1]
	dy = [1,1,0.1,1,1,0.1,1,1]
	dz = [5,1,5,5,1,6,10,2]

	ax1.view_init(30, -45)

	colors = [COLOR_INDY, COLOR_INDY, COLOR_RATIONAL, COLOR_INDY, COLOR_INDY, COLOR_RATIONAL, COLOR_RATIONAL, COLOR_RATIONAL]

	ax1.bar3d(x3, y3, z3, dx, dy, dz, color=colors)



	ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


	ax1.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	
	ax1.grid(False)

	plt.xticks(ticks = [0,1,2,3] ,labels = [r"$C=R$", "", r"$C=U$", ""])
	plt.yticks(ticks = [3,4,5,6] ,labels = ["", r"$H=0$", "", r"$H=1$"])

	ax1.set_zticks([])

	
		# Get rid of the panes
	ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

	# Get rid of the spines
	ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

	plt.show()


def draw_3d_joint_prob_dist_2_x_2(joint_dist=np.array([[4,3],[2,1]]), row_labels=["row 1", "row 2"], col_labels=["col 1", "col 2"]):


	x3 = [0,0,0,2,2,2,4,4]
	y3 = [5,3,2,5,3,2,5,3]
	z3 = np.zeros(8)

	dz = [
		joint_dist[0,0], joint_dist[1,0], joint_dist[0,0] + joint_dist[1,0],
		joint_dist[0,1], joint_dist[1,1], joint_dist[0,1] + joint_dist[1,1],
		joint_dist[0,0] + joint_dist[0,1],
		joint_dist[1,0] + joint_dist[1,1]
		]


	dx = [1,1,1,1,1,1,0.1,0.1]
	dy = [1,1,0.1,1,1,0.1,1,1]

	ax1=plt.gca()

	ax1.view_init(30, -45)

	colors = [COLOR_INDY, COLOR_INDY, COLOR_RATIONAL, COLOR_INDY, COLOR_INDY, COLOR_RATIONAL, COLOR_RATIONAL, COLOR_RATIONAL]

	ax1.bar3d(x3, y3, z3, dx, dy, dz, color=colors)



	ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


	ax1.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	
	ax1.grid(False)

	ylabels = ["", row_labels[0], "", row_labels[1]]	
	xlabels = [col_labels[0], "", col_labels[1], ""]

	plt.xticks(ticks = [0,1,2,3] ,labels = xlabels)
	plt.yticks(ticks = [3,4,5,6] ,labels = ylabels)

	ax1.set_zticks([])

	
		# Get rid of the panes
	ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

	# Get rid of the spines
	ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

	#plt.show()


import numpy as np
import matplotlib.pyplot as plt

def update_priors_given_data(joint_prob_matrix, joint_prob_d_given_h):

	likelihoods = np.multiply(joint_prob_matrix, joint_prob_d_given_h)
	posteriors = likelihoods / np.sum(likelihoods)
	return posteriors


def get_joint_prob_matrix_d_given_h(prob_true_Ann = 0.9, prob_true_Bob = 0.6, X="A"):

	if X == "D1":
		joint_prob_d_given_h = np.array([
			[prob_true_Ann, 1-prob_true_Ann],
			[prob_true_Bob, 1-prob_true_Bob]
			])

	if X == "D2":
		joint_prob_d_given_h = np.array([
			[1-prob_true_Ann, prob_true_Ann],
			[1-prob_true_Bob, prob_true_Bob]
			])	

	if X == "D1D1":
		joint_prob_d_given_h = np.array([
			[prob_true_Ann**2, (1-prob_true_Ann)**2],
			[prob_true_Bob**2, (1-prob_true_Bob)**2]
			])	

	if X == "D2D2":
		joint_prob_d_given_h = np.array([
			[(1-prob_true_Ann)**2, prob_true_Ann**2],
			[(1-prob_true_Bob)**2, prob_true_Bob**2]
			])

	if X == "D1D2":
		joint_prob_d_given_h = np.array([
			[prob_true_Ann*(1-prob_true_Ann), (1-prob_true_Ann)*prob_true_Ann],
			[prob_true_Bob*(1-prob_true_Bob) , (1-prob_true_Bob)*prob_true_Bob]
			])		

	return joint_prob_d_given_h


def get_joint_prior_matrix_independent(prior_H_1, prior_R):

	# All possible hypotheses
	joint_prior_matrix = np.array([
		[prior_H_1*prior_R, (1-prior_H_1)*prior_R],
		[prior_H_1*(1-prior_R), (1-prior_H_1)*(1-prior_R)],
		])
	return joint_prior_matrix


def plot_biased_evaluation():



	fig = plt.figure()
	fig.set_facecolor('white')
	

	prior_H_1 = 0.9
	prior_R = 0.5

	prob_true_Ann = 0.9
	prob_true_Bob = 0.5

	ax1 = fig.add_subplot(221, projection='3d', proj_type = 'ortho')
	ax1.set_facecolor('white') 


	priors = get_joint_prior_matrix_independent(prior_H_1, prior_R)


	draw_3d_joint_prob_dist_2_x_2(joint_dist=priors, row_labels=["R=0", "R=1"], col_labels=["H=1", "H=0"])

	plt.title("Prior")

	# D1
	joint_prob_d1_given_h = get_joint_prob_matrix_d_given_h(prob_true_Ann, prob_true_Bob, "D1")
	post_d1 = update_priors_given_data(priors, joint_prob_d1_given_h)

	ax2 = fig.add_subplot(223, projection='3d', proj_type = 'ortho')
	ax2.set_facecolor('white') 
	
	draw_3d_joint_prob_dist_2_x_2(joint_dist=post_d1, row_labels=["U", "R"], col_labels=["H=1", "H=0"])

	plt.title("Confirming Evidence")
	
	# D0
	joint_prob_d0_given_h = get_joint_prob_matrix_d_given_h(prob_true_Ann, prob_true_Bob, "D2")
	post_d0 = update_priors_given_data(priors, joint_prob_d0_given_h)
	
	ax3 = fig.add_subplot(224, projection='3d', proj_type = 'ortho')
	ax3.set_facecolor('white') 
	draw_3d_joint_prob_dist_2_x_2(joint_dist=post_d0, row_labels=["U", "R"], col_labels=["H=1", "H=0"])
	plt.title("Disconfirming Evidence")



	plt.savefig("images/biased_evaluation_3d.png", dpi=300)

	plt.show()


def draw_table(prob_true_R, prob_true_not_R):

	ax = plt.gca()

	"""
	plt.table(cellText=summary.values,
          rowLabels=summary.index,
          colLabels=summary.columns,
          cellLoc = 'right', rowLoc = 'center',
          loc='right', bbox=[.65,.05,.3,.5])
	"""

	clust_data = np.random.random((10,3))
	table_data = [
		[1,1,prob_true_R],
		[0,1,round(1-prob_true_R,2)],
		[1,0,prob_true_not_R],
		[0,0,1-prob_true_not_R],
		]


	collabel=("H", "R", "P(D=1 | H,R)")
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

	ax.plot(clust_data[:,0],clust_data[:,1])


def draw_bayes_net():


	G = nx.DiGraph()
	G.add_edge("H", "D")
	G.add_edge("R", "D")
	G.add_node(5)
	G.add_node(4)

	# explicitly set positions
	pos = {"H": (-1, 0.5), "R": (1, 0.5), "D": (0, 0), 4:(0,1), 5:(0,-0.5)}

	options = {
	    "font_size": 12,
	    "node_size": 1000,
	    "node_color": "white",
	    "edgecolors": "black",
	    "linewidths": 3,
	    "width": 3,
	}

	nx.draw_networkx(G, pos, **options)

	# Set margins for the axes so that nodes aren't clipped
	ax = plt.gca()
	ax.margins(0.20)
	plt.axis("off")
	



def plot_biased_evaluation_with_bayes_net():


	fig_width, fig_height = plt.gcf().get_size_inches()

	fig = plt.figure(figsize=(fig_width*1.5, fig_height*1.5), constrained_layout=True)
	
	#fig = plt.figure(figsize=(fig_width*2, fig_height), constrained_layout=True)

	# nrows
	gs = fig.add_gridspec(nrows=2, ncols=4)

	prior_H_1 = 0.8
	prior_R = 0.5

	prob_true_Ann = 0.75
	prob_true_Bob = 0.5

	ax1 = fig.add_subplot(gs[0,1:3], projection='3d', proj_type = 'ortho')
	ax1.set_facecolor('white') 


	priors = get_joint_prior_matrix_independent(prior_H_1, prior_R)


	draw_3d_joint_prob_dist_2_x_2(joint_dist=priors, row_labels=["R=0", "R=1"], col_labels=["H=1", "H=0"])

	plt.title("Prior, P(H,R)")

	# D1
	joint_prob_d1_given_h = get_joint_prob_matrix_d_given_h(prob_true_Ann, prob_true_Bob, "D2")
	post_d0 = update_priors_given_data(priors, joint_prob_d1_given_h)

	ax2 = fig.add_subplot(gs[1,:2], projection='3d', proj_type = 'ortho')
	ax2.set_facecolor('white') 
	
	draw_3d_joint_prob_dist_2_x_2(joint_dist=post_d0, row_labels=["R=0", "R=1"], col_labels=["H=1", "H=0"])

	plt.title("Disconfirming Evidence\nP(H,R|D=0)")

	# D0
	joint_prob_d0_given_h = get_joint_prob_matrix_d_given_h(prob_true_Ann, prob_true_Bob, "D1")
	post_d1 = update_priors_given_data(priors, joint_prob_d0_given_h)
	
	ax3 = fig.add_subplot(gs[1,2:], projection='3d', proj_type = 'ortho')
	ax3.set_facecolor('white') 
	draw_3d_joint_prob_dist_2_x_2(joint_dist=post_d1, row_labels=["R=0", "R=1"], col_labels=["H=1", "H=0"])

	plt.title("Confirming Evidence\nP(H,R|D=1)")



	# Condition prob table
	ax5 = plt.subplot(gs[0,3])
	
	draw_table(prob_true_R = prob_true_Ann, prob_true_not_R = prob_true_Bob)


	axs = [ax1, ax2, ax3, ax5]
	ax_labels = ["a", "c", "d", "b"]
	for ax_index in range(4):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		if ax_index < 3:
			ax.text(0.0, 1.0, z=10, s=ax_label, transform=ax.transAxes + trans,
					fontsize='large', va='bottom', weight="bold")
		else:
			ax.text(0.0, 0.8, s=ax_label, transform=ax.transAxes + trans,
					fontsize='large', va='bottom', weight="bold")

	plt.savefig("images/simultaneous_updating_3d_new.png", format="png", dpi=300)

	plt.show()



if __name__=="__main__":
	plot_biased_evaluation_with_bayes_net()