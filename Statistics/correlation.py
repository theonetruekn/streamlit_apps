import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.write("# Pitfalls of Correlation")
st.write("## Why correlation?")
st.write("""
Often in the real world we have two random variables, or basically anything that can be interpreted as a random variable,
and we want to see whether they are related. While there are many types of correlation measures,
Pearson correlation is the most famous one and I would bet you good money to find plenty of correlations
if you open some random Social Science or Economics paper.

As a matter of fact, I would go so far as to say that, if people make it past the abstract, the introduction and the conclusion,
the one thing they will look at is the correlation coefficient $r$. 

There is good reason for this approach as $r$ can be interpreted easily. An $r$ of $1$ means that the two variables are completely related.
If one goes up, the other will also go up (or fall, if $r = -1$).
If $r$ is $0$, then that means that they are independent of each other.

This all sounds great, but there is a catch. Well, not just one, actually, but many. I am here to talk about them.
But first, let's start by understanding what correlation is.
""")
st.write("## What is correlation?")
st.write("""
Correlation is a measure for how much two different random variables are related with each other.
It can be thought of as normalized covariance, but what is covariance?\n

Intuitively, covariance measures how two random variables co-vary, i.e. how changes of magnitude $1$ in the one variable effect the other variable.
Mathematically the covariance of random variables $X$ and $Y$ can be expressed as\n
$$Cov(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])].$$

We then obtain the correlation by normalizing over the product of the standard deviations, i.e.\n 
$$\\rho_{X,Y} = \\frac{Cov(X, Y)}{\sigma_x \sigma_y}$$

We have to divide by the standard deviations as the standard deviations basically give us a sense for how much the variables vary on their own.

If the correlation was calculated from samples of random variables, then we denote the correlation as $r$ instead of $\\rho$.

Lets do a quick experiment, to get some intuition.\n
Let $X = \mathcal{N}(0,1)$ and $Y = 2X + \mathcal{N}(0, \epsilon)$.

Naturally, if $\epsilon = 0$, then $Y = 2X$, so we would expect a covariance of $2$ and a correlation of $1$.\n
Note that the example here is demonstrated via samples drawn from $X$ and $Y$.
Hence, with a large enough sample size $n$, the covariance will indeed approach $2$.

If we increase $\epsilon$, it will overshadow the relationship and the second term of the sum, the *noise*, will dominate, destroying any correlation.

Play with the sliders to get a feeling for this.
""")

noise_level = st.slider('Noise $\epsilon$', 0.0, 10.0, 0.0)
sample_size = st.slider('Sample Size $n$', 1, 10000, 50)

np.random.seed(42)

x = np.random.normal(0, 1, sample_size)
y = 2 * x + np.random.normal(0, noise_level, sample_size)

covariance = np.cov(x, y)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_title(f'$Cov(X, Y) = {covariance:.2f}$, $r_{{XY}} = {correlation:.2f}$')
st.pyplot(fig)

st.write("""
Now that we understand how it works, let's tear it apart.
""")

st.write("## Problems with Correlation")

st.write("### Correlation only captures linear relationships")
st.write("""
The best way to see that correlation only captures linear relationships between two random variables is to check out
the covariance of a non-linear relationship.
""")

st.write("""
Let $X = \mathcal{N}(0,1)$ and $Y = X^2 + \mathcal{N}(0, \epsilon).$
""")

noise_level2 = st.slider('Noise $\epsilon$', 0.0, 10.0, 0.0, key="non-linear-eps")

sample_size_2 = st.slider('Sample Size $n$', 1, 10000, 50, key="non-linear-n")


x_hat = np.random.normal(0, 1, sample_size_2)
y_hat = x_hat**2 + np.random.normal(0, noise_level2, sample_size_2)

covariance = np.cov(x_hat, y_hat)[0, 1]
correlation = np.corrcoef(x_hat, y_hat)[0, 1]

fig, ax = plt.subplots()
ax.scatter(x_hat, y_hat)
ax.set_title(f'$Cov(X, Y) = {covariance:.2f}$, $r_{{XY}} = {correlation:.2f}$')
st.pyplot(fig)

st.write("""
Notice how the correlation does not capture this parabolic relationship? With sufficiently high $n$, $r$ tends to $0$, because the relationship is not linear.

This sucks and it is a major problem with using correlation. Quite frankly, I am not even sure that most economists know this.

There are, however, other measures that can be used to capture non-linear relationships like Mutual Information (blogpost about this coming soon), but they are not used often enough.

Before we move on, I want you to think about the consequences of this. Imagine what can go wrong if the reported
$r$ is inflated or deflated because the underlying relationship that it is trying to describe is non-linear.
""")

st.write("### Correlation is stochastic")

st.write("""
Recall the formula for correlation. Notice that $r$ is a function of two random variables. 
In reality, we always work with samples from random variables, so if we just draw a different sample, we will get a different correlation between the same variables.
Try it out yourself by clicking the `draw again` button.
""")

if 'random_seed' not in st.session_state:
    st.session_state['random_seed'] = 42

if st.button('Draw Again'):
    st.session_state['random_seed'] += 1

noise_level3 = st.slider('Noise $\epsilon$', 0.0, 50.0, 0.0, key="draw_again_eps")
sample_size3 = st.slider('Sample Size $n$', 1, 1000, 30, key="draw_again_n")

np.random.seed(st.session_state['random_seed'])

x = np.random.normal(0, 1, sample_size3)
y = 2 * x + np.random.normal(0, noise_level3, sample_size3)

covariance = np.cov(x, y)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_title(f'$Cov(X, Y) = {covariance:.2f}$, $r_{{XY}} = {correlation:.2f}$')
st.pyplot(fig)

st.write("""
You might be shocked how different the correlation is when drawing from the same distributions.
This does get better with increased sample size, but many papers do not, in fact, have a large enough sample size for this to materialize.

I want you again to ponder the consequences.

How difficult would it be for a malicious actor to engineer the sample to fit his agenda?

Malice aside, how likely is it that at least some papers happen to show a high correlation where there is none, simply due to the sample?

There are some remedies, apart from increasing the sample size, like p-values and confidence intervals,
but they come with their own problems, like the infamous p-hacking (blogpost about this coming soon).

For now, I just want you to be very skeptical whenever you see $r=0.8$ with no additional information.
""")

st.write("### The same $r$ can come from vastly different relationships")

st.write("""
To wrap this up, let's look at four different plots that all have roughly the same correlation.
""")

sns.set_theme(style="ticks")

df = sns.load_dataset("anscombe")

g = sns.lmplot(
    data=df, x="x", y="y", col="dataset", hue="dataset",
    col_wrap=2, palette="muted", ci=None,
    height=4, scatter_kws={"s": 50, "alpha": 1}
)

g.set_titles("Dataset {col_name}")
g.set_axis_labels("X Value", "Y Value")

datasets = df['dataset'].unique()
for i, ax in enumerate(g.axes.flatten()):
    subset = df[df['dataset'] == datasets[i]]
    correlation = subset['x'].corr(subset['y'])
    ax.text(0.05, 0.95, f'Corr: {correlation:.2f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, color='white'))

st.pyplot(g.fig)

st.write("""
This is also known as *Anscombe's Quartet* and it shows, why relying on just $r$ to get a feeling for the relationship is problematic.

This is all to show that one and the same $r$ can come from vastly different relationships. One cure to this, that does work sometimes,
is visualizing the data. Using our mind, we can, for example, see, that Dataset I does have about a linear relationship.
The $r=0.8$ is justified and matches our expectation.

On the other hand, Dataset II clearly follows a non-linear relationship and Dataset III might be totally linear, with one outlier.\n
Detecting outliers is, however, a completely different beast which merits its own blogpost.
""")

st.write("""
# Conclusion
""")
st.write("""
Correlation is a powerful tool to fool people about the existence or non-existence of relationships.\n
Keep this in mind, to not be fooled by others, and especially not by yourself.
""")

st.write("""
___
**Acknowledgements:**\n
I want to give a shout out to Nassim N. Taleb, from whom I learned a lot about statistical fallacies and how to avoid them.
""")