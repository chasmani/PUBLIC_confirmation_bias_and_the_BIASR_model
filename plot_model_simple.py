
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import networkx as nx


def draw_bayes_net(information_dependence=False):

	G = nx.DiGraph()
	G.add_edge("H", "D")
	G.add_edge("R", "D")

	# explicitly set positions
	pos = {"H": (-1, 0.5), "R": (1, 0.5), "D": (0, 0)}

	options = {
	    "font_size": 13,
	    "node_size": 1000,
	    "node_color": "white",
	    "edgecolors": "#404040",
	    "edge_color": ["#404040", "#404040", "grey", "grey"],
	    "linewidths": 4,
	    "width": 4,
		}

	if information_dependence:
		G.add_edge("H", "R")
		G.add_edge("R", "H")
		options["edge_color"] = ["#404040", "grey", "#404040", "grey"]
		options["style"] = ["solid", (0, (3,7)), "solid", (0, (3,7))]


	nx.draw_networkx(G, pos, **options)

	#nx.draw_networkx_nodes(G, pos)
	#nx.draw_networkx_edges(G, pos, edge_color=["red"])


	# Set margins for the axes so that nodes aren't clipped
	ax = plt.gca()
	ax.margins(0.20)
	plt.axis("off")


def draw_arrow_with_text(text=r"$P(H,R|D)$"):

	# draw arrow
	plt.annotate(
	    "",
	    xy=(0.9, 0.45), xycoords="data",
	    xytext=(0.1, 0.45), textcoords="data",
	    arrowprops=dict(
			arrowstyle="-|>", 
		     lw=6, 
		     connectionstyle="arc3",
			 color="#404040"
			 ),
	)

	# draw text
	plt.text(0.5, 0.6, text, size=13, ha="center", va="center",
	         )
	
	ax = plt.gca()
	ax.axis("off")







def draw_bayes_nets_subplots():

	fig = plt.figure(figsize=(12, 2))

	ax1 = plt.subplot(151)

	draw_bayes_net()

	ax2 = plt.subplot(152)

	draw_arrow_with_text(text=r"$P(H,R|D)$")

	ax3 = plt.subplot(153)

	draw_bayes_net(information_dependence=True)

	ax4 = plt.subplot(154)

	draw_arrow_with_text(text=r"$P(H,R) \approx P(H)P(R)$")

	ax5 = plt.subplot(155)

	draw_bayes_net()


	axs = [ax1, ax2, ax3, ax4, ax5]
	ax_labels = ["a", "b", "c", "d", "e"]
	for ax_index in range(5):
		ax = axs[ax_index]
		ax_label = ax_labels[ax_index]

		# label physical distance to the left and up:
		trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
		if ax_index in [0,2,4]:
			ax.text(0.58, 1, s=ax_label, transform=ax.transAxes + trans,
					fontsize=16, va='bottom', weight="bold")
		else:
			ax.text(0.57, 1, s=ax_label, transform=ax.transAxes + trans,
					fontsize=16, va='bottom', weight="bold")

		

	plt.tight_layout()

	plt.savefig("images/bayes_net_simple.png", dpi=300)
	plt.show()
	

if __name__=="__main__":
	draw_bayes_nets_subplots()