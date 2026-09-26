# 24/09/2026

## Linear Regression Notes

Given a dataset, never assume a 1-1 correspondence and overfit the data.

e.g.

$$
x = [1,2,3,4,5]
$$

$$
y = [10,20,30,45,53]
$$

Given $x=4$, $\hat{y}$ is not $45$, as it would imply that the datapoint $(x,y)$ does not consider any other point's influence.

### Linear Model

Linear model = first intuition from the human brain — simplest model.

**A model is what?**

- A rule that takes an input and provides an output.
- Learns from past examples.
- Can identify on unseen data.

### Linear Regression

$$
y = wx + b
$$

- $w$ = weights
- $b$ = bias

---

**Note:** Error function:

$$
(\hat{y} - y)^2 = ((wx+b)-y)^2
$$

which is always an upward parabola $\rightarrow$ it always has a minimum.

Use gradient descent to find the minimum value of error, i.e.

$$
\frac{\partial f}{\partial w}
\quad\text{and}\quad
\frac{\partial f}{\partial b}
$$
