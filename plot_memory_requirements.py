
import numpy as np
import matplotlib.pyplot as plt

COLOR_BIASED = "#2980b9"
COLOR_RATIONAL = "#e67e22"
COLOR_POSITIVE = "#7F8C8D"
COLOR_NEGATIVE = "#E74C3C"

def get_full_posterior_size(k, n):

	return n**int(k)

def get_full_approximation_size(k, n):

	return n*k

def get_blocks_of_10(k, n):

	total = 0
	# Add in blocks of 10, until we get smaller
	while k > 10:
		k -= 10
		total += get_full_posterior_size(10, n)

	# Add the remainder
	total += get_full_posterior_size(k, n)
	return total



def plot_memory_vs_attribute_count():

	n = 2

	ks = np.arange(1,51)

	memory_full_posterior = [get_full_posterior_size(k,n) for k in ks]
	memory_limited_to_10 = [get_blocks_of_10(k,n) for k in ks]
	memory_full_approx = [get_full_approximation_size(k,n) for k in ks]

	plt.plot(ks, memory_full_posterior, label="Full posterior $N_{max}= \infty$", linewidth=2, linestyle="dashed", color=COLOR_RATIONAL)
	plt.plot(ks, memory_limited_to_10, label=r"Partial Approximation $N_{max}=10$", linewidth=2, linestyle="dotted", color=COLOR_POSITIVE)
	plt.plot(ks, memory_full_approx, label="Full approximation $N_{max}=1$", linewidth=2, color=COLOR_BIASED)

	plt.legend()

	plt.xlabel("Number of Attributes, n")
	plt.ylabel("Posterior Memory Space, Cardinality")

	plt.yscale("log")
	plt.savefig("images/memory_space_scaling.png")
	plt.show()

if __name__=="__main__":
	plot_memory_vs_attribute_count()

