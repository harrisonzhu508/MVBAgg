import tensorflow as tf
import gpflow
from gpflow import covariances
from gpflow.posteriors import IndependentPosteriorSingleOutput
from gpflow.config import default_jitter
from gpflow.conditionals.util import base_conditional

class MVBAggPosterior(IndependentPosteriorSingleOutput):
    def __init__(self, num_resolution, **kwargs):
        super().__init__(**kwargs)
        self.num_resolution = num_resolution
        if not isinstance(self.mean_function, gpflow.mean_functions.Zero):
            raise NotImplementedError()
    # could almost be the same as IndependentPosteriorMultiOutput ...
    
    def _conditional_aggregated_fused(
        self, data, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Knn_agg = 0
        Kmn_agg = 0
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        for i in range(self.num_resolution):
            w = data[self.num_resolution + i]
            X = data[self.num_resolution * 2 + i]
            if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
                w = tf.expand_dims(w, axis=0)
                X = tf.expand_dims(X, axis=0)
            Knn = self.kernel.kernels[i].K(X)
            Knn = tf.einsum("bji, bjk -> bk", w, Knn)
            Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

            Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), X)  
            Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_batch x num_inducing x num_data)
            Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

            Kmn_agg += Kmn
            Knn_agg += Knn

        fmean, fvar = base_conditional(
            Kmn_agg, Kmm, Knn_agg, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated(
        self, data, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused(
            data, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_aggregated_fused_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
            w = tf.expand_dims(w, axis=0)
            X = tf.expand_dims(X, axis=0)
        Knn = self.kernel.kernels[i].K(X)
        Knn = tf.einsum("bji, bjk -> bk", w, Knn)
        Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), X)  
        Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_inducing x num_batch)
        Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused_i(
            w, X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def fused_predict_f(
        self, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_fused_i(
        self, Xnew, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        assert len(tf.shape(Xnew)) == 2, "Input array X has to have shape 2 (non-batched)"
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        Knn = self.kernel.kernels[i].K_diag(Xnew)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), Xnew)  

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_f_i(
        self, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused_i(
            X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov



class VBAggPosterior(IndependentPosteriorSingleOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.mean_function, gpflow.mean_functions.Zero):
            raise NotImplementedError()
    # could almost be the same as IndependentPosteriorMultiOutput ...
    
    def _conditional_aggregated_fused(
        self, w, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
            w = tf.expand_dims(w, axis=0)
            X = tf.expand_dims(X, axis=0)
        Knn = self.kernel(X)
        Knn = tf.einsum("bji, bjk -> bk", w, Knn)
        Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

        Kmn = covariances.Kuf(self.X_data, self.kernel, X)
        Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_inducing x num_batch)
        Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated(
        self, w, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused(
            w, X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_aggregated_fused_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
            w = tf.expand_dims(w, axis=0)
            X = tf.expand_dims(X, axis=0)
        Knn = self.kernel.kernels[i].K(X)
        Knn = tf.einsum("bji, bjk -> bk", w, Knn)
        Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), X)  
        Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_inducing x num_batch)
        Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused_i(
            w, X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def fused_predict_f(
        self, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_fused_i(
        self, Xnew, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        assert len(tf.shape(Xnew)) == 2, "Input array X has to have shape 2 (non-batched)"
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        Knn = self.kernel.kernels[i].K_diag(Xnew)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), Xnew)  

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_f_i(
        self, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused_i(
            X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

class GPRNAggPosterior(IndependentPosteriorSingleOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.mean_function, gpflow.mean_functions.Zero):
            raise NotImplementedError()
    # could almost be the same as IndependentPosteriorMultiOutput ...
    
    def _conditional_aggregated_fused(
        self, w, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
            w = tf.expand_dims(w, axis=0)
            X = tf.expand_dims(X, axis=0)
        Knn = self.kernel(X)
        Knn = tf.einsum("bji, bjk -> bk", w, Knn)
        Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

        Kmn = covariances.Kuf(self.X_data, self.kernel, X)
        Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_inducing x num_batch)
        Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated(
        self, w, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused(
            w, X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_aggregated_fused_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """Used in self.fused_predict_f()

        data being a minibatch
        
        """
        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        if len(tf.shape(w)) == 2 and len(tf.shape(X)) == 2:
            w = tf.expand_dims(w, axis=0)
            X = tf.expand_dims(X, axis=0)
        Knn = self.kernel.kernels[i].K(X)
        Knn = tf.einsum("bji, bjk -> bk", w, Knn)
        Knn = tf.einsum("bij, bi -> b", w, Knn) # (num_batch,)

        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), X)  
        Kmn = tf.transpose(Kmn, perm=[1,0,2]) # (num_inducing x num_batch)
        Kmn = tf.einsum("bij, bjk -> ib", Kmn, w)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_aggregated_i(
        self, w, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_aggregated_fused_i(
            w, X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def fused_predict_f(
        self, X, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov

    def _conditional_fused_i(
        self, Xnew, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        assert len(tf.shape(Xnew)) == 2, "Input array X has to have shape 2 (non-batched)"
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        Knn = self.kernel.kernels[i].K_diag(Xnew)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = self.kernel.kernels[i].K(tf.gather(self.X_data.Z, indices=self.kernel.kernels[i].active_dims, axis=1), Xnew)  

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

    def fused_predict_f_i(
        self, X, i, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused_i(
            X, i, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return mean, cov