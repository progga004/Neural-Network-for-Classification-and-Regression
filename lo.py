import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Compute the cross entropy loss."""
    true_label_idx = np.argmax(y_true)
    print(true_label_idx)
    # Softmax computation
    exp_pred = np.exp(y_pred)
    softmax_pred = exp_pred / np.sum(exp_pred)
    
    # Cross entropy loss for this instance
    loss = -np.log(softmax_pred[true_label_idx])
    return loss

# True labels (One-hot encoded)
y = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])

# Predictions
yhat = np.array([[0.5, 0.1, 0.2], [0.3, 0.4, 0.5], [0.9, 0.1, 0.3], [0, 0, 1]])

# Calculate the loss
losses = np.array([cross_entropy_loss(true, pred) for true, pred in zip(y, yhat)])

# Reshape the losses into a column vector
loss_matrix = losses.reshape(-1, 1)

print(loss_matrix)
