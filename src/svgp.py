import tensorflow as tf
from gpflow.kullback_leiblers import gauss_kl
from gpflow.models import (
    SVGP,
    GPModel,
    ExternalDataTrainingLossMixin,
)
from gpflow.models.util import inducingpoint_wrapper
from gpflow import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow import posteriors
from src.posteriors import MVBAggPosterior, VBAggPosterior
import numpy as np


class VBagg(SVGP):
    """
    This is the VBagg
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
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
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = len(self.inducing_variable)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

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

    def compute_sobol(self, df_scaled, list_columns):
        """df_scaled: a dataframe containing all the data used for Sobol estimation

        We compute the first order Sobol index

        """
        # alpha = K_ZZ^{-1} @ m
        alpha = self.posterior().alpha
        Z = self.inducing_variable.Z
        all_features = []
        for list in list_columns:
            all_features += list

        Kmn = self.kernel(df_scaled[all_features].values, Z)
        alpha_j = Kmn @ alpha
        variance_full = (
            np.mean(alpha_j ** 2)
            - np.mean(alpha_j) ** 2
            + self.likelihood.variance.numpy()
        )

        alpha_list = []
        # store posterior means for each component
        # To compute E_{X1,..,Xd}[mean|X_l], we first calculate E_{X_l}[mean^l] for each l
        for i, cols in enumerate(list_columns):
            Z_tmp = tf.gather(Z, indices=self.kernel.kernels[i].active_dims, axis=1)
            Kmn = self.kernel.kernels[i].K(df_scaled[cols].values, Z_tmp)
            alpha_j = Kmn @ alpha
            alpha_list.append(alpha_j)

        sobol = {}
        # compute the 1st order Sobols
        for i in range(len(list_columns)):
            sobol[f"{i}"] = (
                np.mean(alpha_list[i] ** 2) - np.mean(alpha_list[i]) ** 2
            ) / variance_full

        # compute the 2nd order Sobols
        for i in range(len(list_columns)):
            for j in range(i+1, len(list_columns)):
                sobol[f"{i}-{j}"] = (
                    np.mean(
                        2
                        * (alpha_list[i] - np.mean(alpha_list[i]))
                        * (alpha_list[j] - np.mean(alpha_list[j]))
                    )
                    / variance_full
                )

        return sobol, variance_full


class MVBAgg(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the MVBagg
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        num_resolution,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
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
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
        )
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.num_resolution = num_resolution
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        num_inducing = inducing_variable.shape[0]
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def prior_kl(self) -> tf.Tensor:
        """Only implement whitened variational posterior"""
        return gauss_kl(self.q_mu, self.q_sqrt, None)

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        kl = self.prior_kl()
        f_mean, f_var = self.predict_aggregated(data)
        Y = data[-1]
        var_exp = self.likelihood.variational_expectations(
            f_mean,
            f_var,
            Y[:, :, 0],
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(data[0])[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data)

    def predict_aggregated(
        self, data, full_cov: bool = False, full_output_cov: bool = False
    ):
        return self.posterior(
            posteriors.PrecomputeCacheType.NOCACHE
        ).fused_predict_aggregated(
            data, full_cov=full_cov, full_output_cov=full_output_cov
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
        return MVBAggPosterior(
            num_resolution=self.num_resolution,
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

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones(
                    (num_inducing, self.num_latent_gps), dtype=default_float()
                )
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float())
                    for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def compute_sobol(self, df_scaled, list_columns):
        """df_scaled: a dataframe containing all the data used for Sobol estimation

        We compute the first order Sobol index

        """
        # alpha = K_ZZ^{-1} @ m
        alpha = self.posterior().alpha
        Z = self.inducing_variable.Z
        all_features = []
        for list in list_columns:
            all_features += list

        Kmn = self.kernel(df_scaled[all_features].values, Z)
        alpha_j = Kmn @ alpha
        variance_full = (
            np.mean(alpha_j ** 2)
            - np.mean(alpha_j) ** 2
            + self.likelihood.variance.numpy()
        )

        alpha_list = []
        # store posterior means for each component
        # To compute E_{X1,..,Xd}[mean|X_l], we first calculate E_{X_l}[mean^l] for each l
        for i, cols in enumerate(list_columns):
            Z_tmp = tf.gather(Z, indices=self.kernel.kernels[i].active_dims, axis=1)
            Kmn = self.kernel.kernels[i].K(df_scaled[cols].values, Z_tmp)
            alpha_j = Kmn @ alpha
            alpha_list.append(alpha_j)

        sobol = {}
        # compute the 1st order Sobols
        for i in range(len(list_columns)):
            sobol[f"{i}"] = (
                np.mean(alpha_list[i] ** 2) - np.mean(alpha_list[i]) ** 2
            ) / variance_full

        # compute the 2nd order Sobols
        for i in range(len(list_columns)):
            for j in range(i+1, len(list_columns)):
                sobol[f"{i}-{j}"] = (
                    np.mean(
                        2
                        * (alpha_list[i] - np.mean(alpha_list[i]))
                        * (alpha_list[j] - np.mean(alpha_list[j]))
                    )
                    / variance_full
                )

        return sobol, variance_full


class MVBAggBinomial(MVBAgg):
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        kl = self.prior_kl()
        f_mean, f_var = self.predict_aggregated(data)
        counts = data[-2]
        Y = data[-1]
        var_exp = self.likelihood.variational_expectations(
            f_mean, f_var, Y[:, :, 0], counts[:, :, 0]
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(data[0])[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_log_density(
        self,
        f_mean,
        f_var,
        Ynew,
        counts,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        return self.likelihood.predict_log_density(f_mean, f_var, Ynew, counts)


class SVGPBinomial(SVGP):
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        counts = X[:, -1:]
        X = X[:, :-1]
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y, counts)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

class VBAggBinomial(VBagg):
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        N, w, X, counts, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_aggregated(w, X)
        var_exp = self.likelihood.variational_expectations(
            f_mean, f_var, Y[:, :, 0], counts[:, :, 0]
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(data[0])[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_log_density(
        self,
        f_mean,
        f_var,
        Ynew,
        counts,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        return self.likelihood.predict_log_density(f_mean, f_var, Ynew, counts)