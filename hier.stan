data {
  int<lower=0> N;             // Number of time periods
  int<lower=0> M;             // Number of assets
  int<lower=0> K;             // Number of macro factors
  int<lower=0> L;             // Number of asset characteristics
  matrix[N, K] x;             // Macro factors (F), size N x K
  matrix[N * M, L] z;         // Asset characteristics (z), size N*M x L
  matrix[N, M] y;             // Asset returns (R), size N x M
}

parameters {
  real mu;                        // Global intercept
  vector[M] eta_a;                // Random intercepts for assets
  vector[M] eta_b;                // Random coefficients for macro factors
  matrix[M, L] theta_a;           // Coefficients linking z to intercepts (alpha)
  matrix[M, L] theta_b;           // Coefficients linking z to macro sensitivities (beta)
  real<lower=0> sigma;            // Error scale
  vector<lower=-1, upper=1>[M] phi; // Autoregressive coefficients
}

model {
  // Priors
  mu ~ normal(0, 1);
  eta_a ~ normal(0, 1);       // Random intercept priors
  eta_b ~ normal(0, 1);       // Random coefficients for macro factors
  to_vector(theta_a) ~ normal(0, 1);
  to_vector(theta_b) ~ normal(0, 1);
  sigma ~ normal(0, 1);
  phi ~ normal(0, 0.5);

  // Likelihood
  for (n in 2:N) {
    for (i in 1:M) {
      int row = (n - 1) * M + i;  // Reshape index for asset characteristics
      vector[L] z_row = to_vector(z[row]); 

      // Compute intercept alpha and beta
      real alpha_i_t = eta_a[i] + dot_product(z_row, theta_a[i]);
      real beta_i_t = eta_b[i] + dot_product(z_row, theta_b[i]);

      y[n, i] ~ normal(
        mu + alpha_i_t + phi[i] * y[n-1, i] + x[n] * beta_i_t, sigma
      );
    }
  }
}
