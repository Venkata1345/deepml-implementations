import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	X = np.hstack([np.ones((X.shape[0],1)), X])
	m,n = X.shape
	W = np.zeros(n)
	losses = []

	for i in range(iterations):
		z = np.dot(X, W)
		y_hat = 1/(1+np.exp(-z))
		#BCE
		loss = -np.sum(y*np.log(y_hat) + (1-y)* np.log(1-y_hat))
		losses.append(round(float(loss),4))

		# GD
		grad = np.dot((y_hat - y),X)
		W -= learning_rate*grad
	return list(np.round(W,4)), losses
