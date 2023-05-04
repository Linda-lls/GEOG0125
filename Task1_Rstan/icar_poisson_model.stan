functions {
    real icar_normal_lpdf(vector phi, int N, array[] int node1, array[] int node2) {
        return -0.5 * dot_self(phi[node1] - phi[node2]);
    }
}
data {
    int<lower=0> N;
    int<lower=0> N_edges;
    array[N_edges] int<lower=1, upper=N> node1;
    array[N_edges] int<lower=1, upper=N> node2;
    array[N] int<lower=0> Y;                                        // dependent variable i.e., number of deaths in COVID-19
    vector<lower=0>[N] X1;                                          // independent variable1 i.e., overall health index score
    vector<lower=0>[N] X2;                                          // independent variable2 i.e., income score (rate)
    vector<lower=0>[N] X3;                                          // independent variable3 i.e., Living Environment score
    vector<lower=0>[N] E;                                           // estimated number of expected cases of deaths in COVID-19
}
transformed data {
    vector[N] log_offset = log(E);                                  // use the expected cases as an offset and add to the regression model
}
parameters {
    real alpha;                                                     // define the intercept (overall risk in population)
    real beta1;                                                     // define the coefficient1 for the overall health index score variable 
    real beta2;                                                     // define the coefficient2 for the income score (rate) variable
    real beta3;                                                     // define the coefficient3 for the Living Environment score variable
    real<lower=0> sigma;                                            // define the overall standard deviation producted with spatial effect smoothing term phi
    vector[N] phi;                                                  // spatial effect smoothing term or spatial ICAR component of the model 
}
model {
    phi ~ icar_normal(N, node1, node2);                             // prior for the spatial random effects
    Y ~ poisson_log(log_offset + alpha + beta1*X1 +  beta2*X2 +  beta3*X3 + phi*sigma);       // likelihood function i.e., spatial ICAR model using Possion distribution
    alpha ~ normal(0.0, 1.0);                                       // prior for intercept   (weak/uninformative prior)
    beta1 ~ normal(0.0, 1.0);                                       // prior for coefficient1 (weak/uninformative prior)
    beta2 ~ normal(0.0, 1.0);                                       // prior for coefficient2 (weak/uninformative prior)
    beta3 ~ normal(0.0, 1.0);                                       // prior for coefficient3 (weak/uninformative prior)
    sigma ~ normal(0.0, 1.0);                                       // prior for SD          (weak/uninformative prior)
    sum(phi) ~ normal(0, 0.001*N);
}
generated quantities {
    vector[N] eta = alpha + beta1*X1 +  beta2*X2 +  beta3*X3 + phi*sigma;                     // do eta equals alpha + beta1*X1 + beta2*X2 + beta3*X3 + phi*sigma to get the relative risk for areas 
    vector[N] mu = exp(eta);                                        // the exponentiate eta to mu areas-specific relative risk ratios (RRs)
}
