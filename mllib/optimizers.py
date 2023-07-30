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
        super().__init__(
            gradient_fn,
            parameters,
            learning_rate,
            batch_size, 
            **kwargs)  
        self.loss_fn = loss_fn
        self.verbose = verbose


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



class RMSProp(BaseOptimizer):
    def __init__(
            self,
            gradient_fn,
            parameters,
            learning_rate,
            loss_fn,
            decay_rate=0.9,
            epsilon=1e-8,
            batch_size=1,
            verbose=False,
            **kwargs
    ):
        super().__init__(
            gradient_fn,
            parameters,
            learning_rate,
            batch_size,
            **kwargs
        )
        self.loss_fn = loss_fn
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.verbose = verbose
        self.squared_gradients = np.zeros_like(parameters)

    def update_parameters(self, parameters, gradients):
        self.squared_gradients = self.decay_rate * self.squared_gradients + (1 - self.decay_rate) * gradients ** 2
        parameters -= (self.learning_rate / (np.sqrt(self.squared_gradients) + self.epsilon)) * gradients
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
                loss_batch += loss * y_batch.shape[0]

            iteration += 1
            loss_batch /= y.shape[0]
            self.history['loss'].append(loss_batch)

            if prev_loss is None:
                prev_loss = loss_batch
            else:
                error = abs(prev_loss - loss_batch)
                if self.verbose:
                    print(f"{iteration}: prev_loss: {prev_loss:.3f} "
                          f"loss_batch: {loss_batch:.3f} err: {error:.3f}")
                prev_loss = loss_batch

        if iteration == self.max_iter:
            warnings.warn("RMSProp did not converge!")

        return parameters


class Adam(BaseOptimizer):
    """_summary_

    Args:
        BaseOptimizer (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        pass
    
    def update_parameters(self, parameters, gradient):
        pass

    def optimize(self):
        pass


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
    
    rmsprop = RMSProp(
        gradient_fn=gradient_fn,
        parameters=np.array([20.]),
        loss_fn=loss_fn,
        learning_rate=1e-1,
        max_iter=1000,
        tolerance=1e-12
        )
    X = y = np.array([0])
    x_opt = rmsprop.optimize(X, y)
    assert abs(x_opt-3) <= 1e2, "Wrong answer"
    print(f"x_opt: {x_opt} f_opy: {loss_fn(None, None, x_opt)} "
          f"grad_fn: {gradient_fn(None, None, x_opt)}")
