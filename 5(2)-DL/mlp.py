import time
import numpy as np


class MultiLayerPerceptron(object):
    def __init__(
        self,
        nn_input_dim,
        nn_hdim1,
        nn_hdim2,
        nn_hdim3,
        nn_output_dim,
        init="random",
        seed=0,
    ):

        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
            W4: Fourth layer weights
            b4: Fourth layer biases

        Args:
            nn_input_dim: input dimension
            nn_hdim1: number of neurons in hidden layer 1
            nn_hdim2: number of neurons in hidden layer 2
            nn_hdim3: number of neurons in hidden layer 3
            nn_output_dim: output dimension
            init: {'random', 'constant'}
            seed: random seed
        
        Returns:
        """
        # reset seed before start
        np.random.seed(seed)
        self.model = {}

        if init == "random":
            self.model["W1"] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model["b1"] = np.zeros((1, nn_hdim1))
            self.model["W2"] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model["b2"] = np.zeros((1, nn_hdim2))
            self.model["W3"] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model["b3"] = np.zeros((1, nn_hdim3))
            self.model["W4"] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model["b4"] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model["W1"] = np.ones((nn_input_dim, nn_hdim1))
            self.model["b1"] = np.zeros((1, nn_hdim1))
            self.model["W2"] = np.ones((nn_hdim1, nn_hdim2))
            self.model["b2"] = np.zeros((1, nn_hdim2))
            self.model["W3"] = np.ones((nn_hdim2, nn_hdim3))
            self.model["b3"] = np.zeros((1, nn_hdim3))
            self.model["W4"] = np.ones((nn_hdim3, nn_output_dim))
            self.model["b4"] = np.zeros((1, nn_output_dim))
        else:
            raise ValueError("init must be 'random' or 'constant'")

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
                
        Args:
            X: Input data of shape (N, D)
                    
        Returns:
            y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
        """
        W1, b1 = self.model["W1"], self.model["b1"]
        W2, b2 = self.model["W2"], self.model["b2"]
        W3, b3 = self.model["W3"], self.model["b3"]
        W4, b4 = self.model["W4"], self.model["b4"]

        ### CODE HERE ###
        # 행렬 연산 후 activation function 적용
        # y_hat을 X.shape[0] 크기로 reshape해야한다.
        h1 = X @ W1 + b1
        z1 = relu(h1)

        h2 = z1 @ W2 + b2
        z2 = leakyrelu(h2)

        h3 = z2 @ W3 + b3
        z3 = tanh(h3 + h1)

        h4 = z3 @ W4 + b4
        y_hat = sigmoid(h4)
        y_hat = y_hat.reshape(-1)
        #################

        assert y_hat.shape == (X.shape[0],), (
            f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(X.shape[0],)}"
        )

        cache = {
            "h1": h1, "z1": z1,
            "h2": h2, "z2": z2,
            "h3": h3, "z3": z3,
            "h4": h4, "y_hat": y_hat,
        }
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute gradients for all parameters.

        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) Training labels (N, ) -> (N, 1)
            L2_norm: (int) L2 normalization coefficient
                    
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        """
        W1, W2, W3, W4 = (
            self.model["W1"],
            self.model["W2"],
            self.model["W3"],
            self.model["W4"],
        )
        h1, z1 = cache["h1"], cache["z1"]
        h2, z2 = cache["h2"], cache["z2"]
        h3, z3 = cache["h3"], cache["z3"]
        y_hat = cache["y_hat"]

        # For matrix computation

        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)

        ############################################################
        # gradient 계산하는 과정

        dy_hat = (y_hat - y) / (y_hat * (1 - y_hat) + 1e-15)

        dh4 = dy_hat * y_hat * (1 - y_hat)
        db4 = np.sum(dh4, axis=0, keepdims=True)
        dW4 = z3.T @ dh4 + 2 * L2_norm * W4
        dz3 = dh4 @ W4.T

        ### CODE HERE ###
        # 'gradient 계산하는 과정'을 참고하여 gradient 계산
        # dh3, db3, dW3, dz2
        # dh2, db2, dW2, dz1
        # dh1, db1, dW1
        dh3 = dz3 * (1 - z3 ** 2)
        db3 = np.sum(dh3, axis=0, keepdims=True)
        dW3 = z2.T @ dh3 + 2 * L2_norm * W3
        dz2 = dh3 @ W3.T

        dh2 = dz2 * leakyrelu_grad(h2)
        db2 = np.sum(dh2, axis=0, keepdims=True)
        dW2 = z1.T @ dh2 + 2 * L2_norm * W2
        dz1 = dh2 @ W2.T

        # h1은 ReLU 경로뿐 아니라 z3 = tanh(h3 + h1)의 skip
        # connection에도 직접 연결되어 있으므로 두 gradient를 더한다.
        dh1 = dz1 * relu_grad(h1) + dh3
        db1 = np.sum(dh1, axis=0, keepdims=True)
        dW1 = X.T @ dh1 + 2 * L2_norm * W1
        #################
        ############################################################

        grads = {
            "dy_hat": dy_hat,
            "dh4": dh4,
            "dW4": dW4,
            "db4": db4,
            "dW3": dW3,
            "db3": db3,
            "dW2": dW2,
            "db2": db2,
            "dW1": dW1,
            "db1": db1,
        }
        return grads

    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions: Evaluate the total loss on the dataset
                
        Args:
            y_pred: (numpy array) Predicted target (N,)
            y_true: (numpy array) Array of training labels (N,)
                
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.BCE loss + L2 regularization.
        """
        W1, W2, W3, W4 = (
            self.model["W1"],
            self.model["W2"],
            self.model["W3"],
            self.model["W4"],
        )

        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        log_loss = -np.sum(y_true * np.log(y_pred)+ (1 - y_true) * np.log(1 - y_pred))
        l2_loss = L2_norm * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2) + np.sum(W4 ** 2))
        total_loss = log_loss + l2_loss

        return total_loss

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        learning_rate=1e-3,
        L2_norm=0.0,
        epoch=20000,
        print_loss=True,
        optimizer="bgd",
        batch_size=32,
        seed=0,
        eval_every=10,
    ):
        """
        Train with BGD, SGD, or Mini-batch GD.

        optimizer:
            'bgd'       -> full batch
            'sgd'       -> batch size 1
            'minibatch' -> user-defined batch_size
        """
        if optimizer not in {"bgd", "sgd", "minibatch"}:
            raise ValueError("optimizer must be 'bgd', 'sgd', or 'minibatch'")

        n = X_train.shape[0]

        if optimizer == "bgd":
            effective_batch_size = n
        elif optimizer == "sgd":
            effective_batch_size = 1
        else:
            if batch_size is None or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            effective_batch_size = min(batch_size, n)

        rng = np.random.default_rng(seed)

        history = {
            "epoch": [],
            "update_step": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "elapsed_sec": [],
            # backward-compatible keys
            "loss_history": [],
            "train_acc_history": [],
            "val_acc_history": [],
        }

        update_step = 0
        start_time = time.perf_counter()

        for it in range(epoch):
            indices = rng.permutation(n)

            for start in range(0, n, effective_batch_size):
                idx = indices[start:start + effective_batch_size]
                X_batch = X_train[idx]
                y_batch = y_train[idx]

                ### CODE HERE ###
                # 1. forward propagation
                # 2. loss 계산
                # 3. back propagation 수행 후 gradient update
                y_hat, cache = self.forward_propagation(X_batch)
                loss = self.compute_loss(y_hat, y_batch, L2_norm)
                grad = self.back_propagation(
                    cache, X_batch, y_batch, L2_norm
                )

                # Gradient update
                self.model["W1"] -= learning_rate * grad["dW1"]
                self.model["b1"] -= learning_rate * grad["db1"]
                self.model["W2"] -= learning_rate * grad["dW2"]
                self.model["b2"] -= learning_rate * grad["db2"]
                self.model["W3"] -= learning_rate * grad["dW3"]
                self.model["b3"] -= learning_rate * grad["db3"]
                self.model["W4"] -= learning_rate * grad["dW4"]
                self.model["b4"] -= learning_rate * grad["db4"]
                #################

                update_step += 1

            if (it + 1) % eval_every == 0 or (it + 1) == epoch:
                train_prob, _ = self.forward_propagation(X_train)
                train_loss = self.compute_loss(train_prob, y_train, L2_norm)
                train_pred = self.predict(X_train)
                train_acc = np.mean(train_pred == y_train)

                history["epoch"].append(it + 1)
                history["update_step"].append(update_step)
                history["train_loss"].append(float(train_loss))
                history["train_acc"].append(float(train_acc))
                history["loss_history"].append(float(train_loss))
                history["train_acc_history"].append(float(train_acc))
                history["elapsed_sec"].append(time.perf_counter() - start_time)

                if X_val is not None and y_val is not None:
                    val_prob, _ = self.forward_propagation(X_val)
                    val_loss = self.compute_loss(val_prob, y_val, L2_norm=0.0)
                    val_pred = self.predict(X_val)
                    val_acc = np.mean(val_pred == y_val)

                    history["val_loss"].append(float(val_loss))
                    history["val_acc"].append(float(val_acc))
                    history["val_acc_history"].append(float(val_acc))

                if print_loss and (
                    (it + 1) % max(eval_every * 10, 1) == 0
                    or (it + 1) == epoch
                ):
                    msg = (
                        f"[{optimizer}] epoch={it + 1}, "
                        f"train_loss={train_loss:.4f}, "
                        f"train_acc={train_acc:.3f}"
                    )
                    if X_val is not None and y_val is not None:
                        msg += (
                            f", val_loss={val_loss:.4f}, "
                            f"val_acc={val_acc:.3f}"
                        )
                    print(msg)

        return history

    def predict(self, X):
        ### CODE HERE ###
        # Binary classification: 0.5 이상이면 1, 아니면 0
        probabilities, _ = self.forward_propagation(X)
        predictions = (probabilities >= 0.5).astype(int)
        #################
        return predictions


def tanh(x):
    return np.tanh(x)


def relu(x):
    ### CODE HERE ###
    x = np.maximum(0, x)
    #################
    return x


def relu_grad(x):
    ### CODE HERE ###
    grad = (x > 0).astype(float)
    #################
    return grad


def leakyrelu(x, alpha=0.01):
    ### CODE HERE ###
    x = np.where(x > 0, x, alpha * x)
    #################
    return x


def leakyrelu_grad(x, alpha=0.01):
    ### CODE HERE ###
    grad = np.where(x > 0, 1.0, alpha)
    #################
    return grad


def sigmoid(x):
    ### CODE HERE ###
    x = np.clip(x, -500, 500)
    x = 1 / (1 + np.exp(-x))
    #################
    return x
