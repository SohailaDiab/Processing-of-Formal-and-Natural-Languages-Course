Let's break down where the derivatives come from in the Backward Pass of Backpropagation Through Time (BPTT) for Recurrent Neural Networks (RNNs). We will use the image as a reference to explain the steps.

### Key Components in Backpropagation
- $\[ E_t \]$: The loss at time step $\[ t \]$.
- $\[ y_t \]$: The true label at time step $\[ t \]$.
- $\[ \hat{y}_t \]$: The predicted output at time step $\[ t \]$.
- $\[ O_t \]$: The raw output before applying the softmax.
- $\[ s_t \]$: The hidden state at time step $\[ t \]$.
- $\[ x_t \]$: The input at time step $\[ t \]$.

### Forward Pass Review
1. **Hidden State**: $\[ s_t = \tanh(W s_{t-1} + U x_t) \]$
2. **Output**: $\[ O_t = V s_t \]$
3. **Prediction**: $\[ \hat{y}_t = \text{softmax}(O_t) \]$

### Loss Function
The cross-entropy loss for a single time step $\[ t \]$:
$\[ E_t = - y_t \log(\hat{y}_t) \]$

### Backward Pass (BPTT)
We need to compute the gradients of the loss with respect to the weight matrices $\[ W \]$, $\[ U \]$, and $\[ V \]$.

#### Step-by-Step Derivatives
Let's start with the gradients of $\[ E_t \]$ with respect to $\[ V \]$.

#### 1. Gradient with respect to $\[ V \]$
The loss depends on $\[ V \]$ through the output $\[ O_t \]$:
$\[ \frac{\partial E_t}{\partial V} = \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial O_t} \cdot \frac{\partial O_t}{\partial V} \]$

- **Derivative of the loss with respect to the predicted output**:
  $\[ \frac{\partial E_t}{\partial \hat{y}_t} = \hat{y}_t - y_t \]$

- **Derivative of the softmax function**:
  $\[ \frac{\partial \hat{y}_t}{\partial O_t} \]$
  This involves the Jacobian matrix of the softmax function.

- **Derivative of $\[ O_t \]$ with respect to $\[ V \]$**:
  $\[ \frac{\partial O_t}{\partial V} = s_t \]$

Combining these:
$\[ \frac{\partial E_t}{\partial V} = (\hat{y}_t - y_t) s_t \]$

#### 2. Gradient with respect to $\[ W \]$ and $\[ U \]$
These weights affect the hidden states, so we need to use the chain rule through the hidden states:

$\[ \frac{\partial E_t}{\partial W} = \frac{\partial E_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial W} \]$

Since $\[ s_t \]$ depends on previous hidden states, we have to consider the chain of dependencies back through time. Let's look at the derivative $\[ \frac{\partial s_t}{\partial W} \]$:

$\[ s_t = \tanh(W s_{t-1} + U x_t) \]$

- **Derivative of the loss with respect to the hidden state**:
  $\[ \frac{\partial E_t}{\partial s_t} \]$
  This involves summing the contributions from future time steps (backpropagating through time).

- **Derivative of $\[ s_t \]$ with respect to $\[ W \]$**:
  $\[ \frac{\partial s_t}{\partial W} = (1 - s_t^2) s_{t-1} \]$
  Here, $\[ 1 - s_t^2 \]$ is the derivative of the $\[ \tanh \]$ function.

The total gradient accumulates contributions from all time steps:
$\[ \frac{\partial E}{\partial W} = \sum_{t=0}^T \sum_{k=t}^T \frac{\partial E_k}{\partial s_k} \cdot \frac{\partial s_k}{\partial s_t} \cdot \frac{\partial s_t}{\partial W} \]$

#### Visualization of Gradients in the Image
- The top part of the image explains the recursive dependency of $\[ \frac{\partial s_t}{\partial W} \]$:
  $\[ \frac{\partial s_3}{\partial W} = \sum_k \frac{\partial s_3}{\partial s_k} \frac{\partial s_k}{\partial W} \]$

- The middle part shows a similar approach for $\[ \frac{\partial s_3}{\partial U} \]$:
  $\[ \frac{\partial s_3}{\partial U} = \sum_k \frac{\partial s_3}{\partial s_k} \frac{\partial s_k}{\partial U} \]$

- The bottom part emphasizes summing up gradients at each time step:
  $\[ \frac{\partial E}{\partial V} = \sum_t \frac{\partial E_t}{\partial V} \]$
  $\[ \frac{\partial E}{\partial W} = \sum_t \sum_k \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial O_t} \cdot \frac{\partial O_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial s_k} \cdot \frac{\partial s_k}{\partial W} \]$
  $\[ \frac{\partial E}{\partial U} = \sum_t \sum_k \frac{\partial E_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial O_t} \cdot \frac{\partial O_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial s_k} \cdot \frac{\partial s_k}{\partial U} \]$

### Summary
- **Gradients with respect to $\[ V \]$** are straightforward since they directly influence the output.
- **Gradients with respect to $\[ W \]$ and $\[ U \]$** require considering dependencies back through time due to the recurrent nature of RNNs.
- **The chain rule is applied recursively** to account for these dependencies, leading to the summation of gradients from all relevant time steps.

BPTT involves these steps to ensure that the model learns by properly adjusting all weight matrices, capturing the temporal dependencies in sequential data.
