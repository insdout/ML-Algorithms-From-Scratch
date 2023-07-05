from base import BaseOptimizer
import numpy as np
import warnings

"https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/optimizers.py"
"https://towardsdatascience.com/linear-regression-and-gradient-descent-using-only-numpy-53104a834f75"
"https://ml-lectures.org/docs/supervised_learning_wo_NNs/Linear-regression.htmlg"


class SGD(BaseOptimizer):
    def __init__(
            self,
            gradient_fn,
            parameters,
            learning_rate,
            loss_fn,
            batch_size=1,
            verbose=False,
            **kwargs
    ):
        super().__init__(gradient_fn, parameters, learning_rate, **kwargs)
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.verbose = verbose

    def batch_generator(self, X, y):
        n_samples = X.shape[0]
        batch_size = self.batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]

    def update_parameters(self, parameters, gradients):
        parameters -= self.learning_rate * gradients
        return parameters

    def optimize(self, X, y):
        iteration = 0
        parameters = self.parameters
        error = float("inf")
        prev_loss = None

        while iteration < self.max_iter and error > self.tol:
            loss_batch = 0
            for batch in self.batch_generator(X, y):
                X_batch, y_batch = batch
                gradients = self.gradient_fn(X_batch, y_batch, parameters)

                parameters = self.update_parameters(parameters, gradients)
                self.history["parameters"].append(parameters)
                loss = self.loss_fn(X_batch, y_batch, parameters)

                # Calculate error
                loss_batch += loss*y_batch.shape[0]
            iteration += 1
            loss_batch /= y.shape[0]
            self.history['loss'].append(loss_batch)
            if prev_loss is None:
                prev_loss = loss_batch
            else:
                error = abs(prev_loss - loss_batch)
                if self.verbose:
                    print(f"{iteration}: prev_loss: {prev_loss :.3f} "
                          f"loss_batch: {loss_batch :.3f} err: {error :.3f}")
                prev_loss = loss_batch

        if iteration == self.max_iter:
            warnings.warn("SGD did not converge!")
        return parameters


if __name__ == "__main__":
    def loss_fn(z, y, x):
        return x**2 - 6*x + 9

    def gradient_fn(z, y, x):
        return 2*x - 6

    sgd = SGD(
        gradient_fn=gradient_fn,
        parameters=np.array([20.]),
        loss_fn=loss_fn,
        learning_rate=1e-2,
        max_iter=1000,
        tolerance=1e-12
    )
    X = y = np.array([0])
    x_opt = sgd.optimize(X, y)
    assert abs(x_opt-3) <= 1e2, "Wrong answer"
    print(f"x_opt: {x_opt} f_opy: {loss_fn(None, None, x_opt)} "
          f"grad_fn: {gradient_fn(None, None, x_opt)}")

