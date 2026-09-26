import numpy as np
import time
x = np.array([1.,2.,3.,4.,5.])
y = np.array([32.,41.,55.,60.,74])

w,b = 132,-30
lr = 0.01
prev_loss = float('inf')
start = time.perf_counter()
for step in range(3001):
  y_hat = w*x + b
  err = y_hat - y
  loss = np.mean(err**2);
  dw = 2*np.mean(err*x)
  db = 2*np.mean(err)
  w -= lr*dw
  b -= lr*db
  if(prev_loss < loss):
    lr*=0.1
    prev_loss = loss
  if step % 500 == 0:
    print(f"step {step:4d} loss {loss:8.2f} w {w:6.2f} b {b:6.2f}")
end = time.perf_counter()
runtime = end-start
print(f"runtime = {runtime:.9f} seconds") #Runtime in C is = 0.000226000 seconds; In python it is around 0.12 seconds
