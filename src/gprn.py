import tensorflow as tf
from gpflow.models import (
    SVGP,
)
from gpflow.models.util import inducingpoint_wrapper
from gpflow import posteriors
from gpflow import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from src.posteriors import VBAggPosterior
import numpy as np

class GPRNAgg(SVGP):
    """
    This is the GPRN version of VBAgg
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        mean_function=None,
        num_latent_gps: int = 1,
        whiten: bool = True,
        num_data=None,
    ):
        """
        Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
        """
        # init the super class, accept args
        super().__init__(
            kernel,
            likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
        )
        self.num_data = num_data
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = len(self.inducing_variable)
        self._init_variational_parameters(num_inducing)

    def _init_variational_parameters(self, num_inducing):
        """constructs q(u) for both the variational GP weights and GP regressor

        Weight GP mean and q_sqrt have postfixs _weights
        
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps))
        q_mu_weights = np.zeros((num_inducing, self.num_latent_gps)) 
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]
        self.q_mu_weights = Parameter(q_mu_weights, dtype=default_float())  # [M, P]

        q_sqrt = [
            np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
        ]
        q_sqrt = np.array(q_sqrt)
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]

        q_sqrt_weights = [
            np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
        ]
        q_sqrt_weights = np.array(q_sqrt_weights)
        self.q_sqrt_weights = Parameter(q_sqrt_weights, transform=triangular())  # [P, M, M]

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        N, w, X, Y = data
        kl = self.prior_kl()

        f_mean, f_var = self.predict_aggregated(w, X)
        var_exp = self.likelihood.variational_expectations(
            f_mean,
            f_var,
            Y[:, :, 0],
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_aggregated(
        self, w, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        return self.posterior(
            posteriors.PrecomputeCacheType.NOCACHE
        ).fused_predict_aggregated(
            w, X, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def predict_aggregated_i(
        self,
        w: np.ndarray,
        X: np.ndarray,
        i: int,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ):
        return self.posterior(
            posteriors.PrecomputeCacheType.NOCACHE
        ).fused_predict_aggregated_i(
            w, X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        assert (
            tf.shape(Xnew)[-1] == self.inducing_variable.Z.shape[1]
        ), f"Input X has to have last dimension {self.inducing_variable.Z.shape[1]}"
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def predict_f_i(
        self, Xnew: np.ndarray, i: int, full_cov=False, full_output_cov=False
    ):
        assert tf.shape(Xnew)[-1] == len(
            self.kernel.kernels[i].active_dims
        ), f"Input X has to have last dimension {len(self.kernel.kernels[i].active_dims)}"
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f_i(
            Xnew, i, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def posterior(self, precompute_cache=posteriors.PrecomputeCacheType.TENSOR):
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """
        return VBAggPosterior(
            kernel=self.kernel,
            inducing_variable=self.inducing_variable,
            q_mu=self.q_mu,
            q_sqrt=self.q_sqrt,
            whiten=self.whiten,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_log_density(
        self, f_mean, f_var, Ynew, full_cov: bool = False, full_output_cov: bool = False
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        return self.likelihood.predict_log_density(f_mean, f_var, Ynew)