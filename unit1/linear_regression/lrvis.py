import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Prepare data for animation by re-running the gradient descent and storing history
w_history = []
b_history = []
loss_history = []

w,b = 0.0,0.0
lr = 0.01

for step in range(3001):
  y_hat = w*x + b
  err = y_hat - y
  loss = np.mean(err**2);
  dw = 2*np.mean(err*x)
  db = 2*np.mean(err)
  w -= lr*dw
  b -= lr*db

  w_history.append(w)
  b_history.append(b)
  loss_history.append(loss)

  if step % 500 == 0:
    print(f"step {step:4d} loss {loss:8.2f} w {w:6.2f} b {b:6.2f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot of the data points
ax1.scatter(x, y, color='red', label='Data Points')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gradient Descent: Regression Line Evolution')
ax1.set_xlim(np.min(x) - 1, np.max(x) + 1)
ax1.set_ylim(np.min(y) - 10, np.max(y) + 10)

# Initialize the regression line
line, = ax1.plot([], [], color='blue', label='Regression Line')

# Plot of the loss function
ax2.plot(loss_history, color='green')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Loss')
ax2.set_title('Loss over Steps')
ax2.grid(True)

# Text for displaying current w, b, loss
text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10)

def update(frame):
    current_w = w_history[frame]
    current_b = b_history[frame]
    y_pred = current_w * x + current_b
    line.set_data(x, y_pred)
    text.set_text(f'Step: {frame}\nw: {current_w:.2f}\nb: {current_b:.2f}\nLoss: {loss_history[frame]:.2f}')
    return line, text

animation_frames = range(0, len(w_history), 50) # Animate every 50 steps for smoother visualization
ani = animation.FuncAnimation(
    fig,
    update,
    frames=animation_frames,
    blit=True,
    interval=50  # milliseconds between frames
)

plt.tight_layout()
plt.show()

# To save the animation, you might need ffmpeg installed:
# ani.save('gradient_descent_animation.gif', writer='pillow', fps=20)
