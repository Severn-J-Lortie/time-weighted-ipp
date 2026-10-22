using AbstractGPs
using Plots
using Random
using LinearAlgebra

x = [
    0.8658165855998895,
    0.6661700880180962,
    0.8049218148148531,
    0.7714303440386239,
    0.14790478354654835,
    0.8666105548197428,
    0.007044577166530286,
    0.026331737288148638,
    0.17188596617099916,
    0.8897812990554013,
    0.24323574561119998,
    0.028590102134105955,
]
y = [
    1.5255314337144372,
    3.6434202968230003,
    3.010885733911661,
    3.774442382979625,
    3.3687639483798324,
    1.5506452040608503,
    3.790447985799683,
    3.8689707574953,
    3.4933565751758713,
    1.4284538820635841,
    3.8715350915692364,
    3.7045949061144983,
]
x_train = x[1:8]
y_train = y[1:8]
x_test = x[9:end]
y_test = y[9:end]

f = GP(Matern52Kernel()) # f is now the prior GP
noise_var = 0.1
fx = f(x_train, noise_var) # basically, evaluate the prior GP at x_train (this constructs a prior covariance matrix of those 8 points)
p_fx = posterior(fx, y_train) # condition the GP on observed values. This updates the mean and covariance to better fit the distribution of the training data

prior_cov = cov(fx)
posterior_cov = cov(p_fx(x_train, noise_var))

scatter(
    x_train,
    y_train;
    xlim=(0, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (default parameters)",
    label="Train Data",
)
scatter!(x_test, y_test; label="Test Data")
p = plot!(0:0.001:1, p_fx; label=false, ribbon_scale=2)