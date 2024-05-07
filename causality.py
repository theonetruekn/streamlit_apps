import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score

st.title('Getting from Correlation to Causation - kind of')

st.write("""
You have probably by now heard someone say with a smug look of superiority: "Actually, correlation does not equal causation".
This is technically true. *Causation* itself is not well-defined and has deep philosophical implications, going back to Hume at the very least.

I want to take you on a journey down the rabbit hole of a certain type of causation, namely 'Granger causality'.

Let's start by understanding what correlation is.
""")

st.write("""
# From Covariance to Correlation
""")

st.write("""
Correlation can be thought of as a normalized covariance, but what is covariance?
Intuitively, covariance measures how two random variables co-vary, i.e. how changes of magnitude $1$ in the one variable effect the other variable.
Mathematically the covariance of random variables $X$ and $Y$ can be expressed as\n
$$Cov(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])].$$

We then obtain the correlation by normalizing over the product of the standard deviations, i.e.\n 
$$\\rho_{X,Y} = \\frac{Cov(X, Y)}{\sigma_x \sigma_y}$$

We do this as the standard deviations basically give us a sense for how much the variables vary on their own.

Lets do a quick experiment, to get some intuition.\n
Let $X = \mathcal{N}(0,1)$ and $Y = 2~X + \mathcal{N}(0, \epsilon)$.

Naturally, if $\epsilon = 0$, then $Y = 2~X$, so we would expect a covariance of $2$ and a correlation of $1$.\n
Note that the example here is demonstrated via samples drawn from $X$ and $Y$.
Hence, with a large enough sample size $n$, the covariance will indeed approach $2$.

If we increase $\epsilon$, it will overshadow the relationship and the second term of the sum, the *noise*, will dominate, destroying any correlation.
""")

noise_level = st.slider('Noise $\epsilon$', 0.0, 50.0, 0.0)
sample_size = st.slider('Sample Size $n$', 1, 100000, 50)

np.random.seed(42)

x = np.random.normal(0, 1, sample_size)
y = 2 * x + np.random.normal(0, noise_level, sample_size)

covariance = np.cov(x, y)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_title(f'Covariance: {covariance:.2f}, Correlation: {correlation:.2f}')
st.pyplot(fig)

st.write("""
Great, now that we got that out of the way, let's try the same spiel with a non-linear relationship.
""")
st.write("""
## Correlation for non-linear Relationships
""")
st.write("""
Let $X = \mathcal{N}(0,1)$ and $Y = X^2 + \mathcal{N}(0, \epsilon).$
""")

sample_size_2 = st.slider('Sample Size $n$ for non-linear relationship', 1, 100000, 50)


x_hat = np.random.normal(0, 1, sample_size_2)
y_hat = x_hat**2 + np.random.normal(0, noise_level, sample_size_2)

covariance = np.cov(x_hat, y_hat)[0, 1]
correlation = np.corrcoef(x_hat, y_hat)[0, 1]

fig, ax = plt.subplots()
ax.scatter(x_hat, y_hat)
ax.set_title(f'Covariance: {covariance:.2f}, Correlation: {correlation:.2f}')
st.pyplot(fig)

st.write("""
This sucks and it is a major problem with using correlation - it only captures linear relationships.

Luckily for us, there are some other measures that can be used to capture non-linear relationships, making use of Information Theory.
""")

st.write("""
# Mutual Information
""")

st.write("""
*Mutual Information* is an information-theoretic measures that is based on the *Shannon entropy* $H$.
The entropy of a random variable $X$ is given by\n 
$H(X) = \int_{x \in X}~p(x)~log(p(x)) dx.$
The mutual information is then given by\n 
$I(X;Y) = H(X) + H(Y) - H(X,Y)$. \n
Note that the maximum mutual information is given by $I_{\\text{max}}(X;Y) = \\text{min}~(H(x), H(y))$.

To not have to work with this nasty integral, we discretize our samples using the Freedman-Diaconis rule.

""")

def calculate_entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
    return entropy

x_binned = np.digitize(x_hat, bins=np.histogram_bin_edges(x_hat, bins='fd'))
y_binned = np.digitize(y_hat, bins=np.histogram_bin_edges(y_hat, bins='fd'))

mi = mutual_info_score(x_binned, y_binned)

entropy_x = calculate_entropy(x_binned)
entropy_y = calculate_entropy(y_binned)

st.write(f"""
For our discretized X and Y we get a mutual information of $I(X;Y) = {mi:.3f}$.
Sadly, we lose some interpretability like with correlation, as mutual information can be arbitrarily large.
We can, however, normalize it with the maximum possible mutual information. Our normalized mutual information is $\\hat{{I}}(X;Y)={mi/min(entropy_x, entropy_y):.3f}$.

Let's keep the existence of mutual information in the back of our minds for now. It will come in handy later.
""")

st.write("""
## Adding the dimension of Time
""")

st.write("""
Often, when discussing causality, we assume that time is linear and the *cause* must always come before the *effect*.
With this assumption in mind, it makes sense to move from random variables to random processes as our main object of consideration. 
A random process is basically a random variable indexed over time.\n
We call a random process a *time series process*, if the index set are the natural numbers.\n
A *time series* is then the realization of an underlying time series process.

Adding the the dimension of time allows us to ask questions about seasonality, general trends over time and whether there is a
correlation between different points in time of our time series process. This is called *autocorrelation* and can be quite useful, so let's dig a bit into it.

Let ${X_t}$ be a time series process and $X_t$ the realization of that time series process at time $t$.
Then, the autocorrelation at times $t_1$ and $t_2$ are simply the correlation 
$\\rho_{XX} = \\frac{Cov(X_{t_1}, X_{t_2})}{\sigma_{X_{t_1}}\sigma_{X_{t_2}}}$.

This looks scary, but it becomes more understandable when we look at some plots.
""")

n_points = 100

time_index = np.arange(n_points)

frequency = st.slider('Frequency', 1, 100, 10)
amplitude = st.slider('Amplitude', 1, 10, 5)
cyclical_component = amplitude * np.sin(2 * np.pi * frequency * time_index / n_points)

noise_scale = 1.0
noise = np.random.normal(loc=0.0, scale=noise_scale, size=n_points)

time_series = cyclical_component + noise
lag = frequency
autocorrelation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]

plt.figure(figsize=(10, 5))
plt.plot(time_index, time_series, markersize=5)
plt.title('Realization of a discrete Time Series')
plt.xlabel('Time t')

st.pyplot(plt)

st.write(f"""
The plot above shows how a discrete time series can be plotted. As you can see, it is clearly a noisy sine function.
The autocorrelation of $X_t$ and $X_{{t+{frequency}}}$ hence approaches $1$ if we increase the sample size.\n
The sample autocorrelation is ${autocorrelation: .3f}$.

You can play with the lag and see how the autocorrelation changes.
""")
st.write("""
**What happens if the lag is a multiple of the frequency?**\n
**What happens if the lag is a multiple of half the frequency?**
""")

lag = st.slider('Lag', 1, n_points - 1, 10)

if lag < len(time_series):
    autocorrelation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
else:
    autocorrelation = 'undefined'

lagged_series = np.roll(time_series, lag)
lagged_series[:lag] = np.nan

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_index, time_series, label='Original Series')
ax.plot(time_index, lagged_series, label=f'Lagged Series by {lag} steps')
ax.set_xlabel('Time t')
ax.legend()

st.pyplot(plt)

st.write(f"The autocorrelation at lag {lag} is: {autocorrelation:.3f}")

st.write("# Roadmap")

st.write("""
- Auto-Regression
- Granger Causality
- Transfer Entropy
""")