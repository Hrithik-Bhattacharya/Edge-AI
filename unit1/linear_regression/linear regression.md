_24/09/2026_
**Linear Regression Notes**
Given a dataset, never assume a 1-1 correspondence and overfit the data.
e.g. $x = _[1,2,3,4,5]_   $y= _[10,20,30,45,53]_ 
given $x=4, $y<sub>pred</sub> is not 45 as it would imply that datapoint ($x , $y) does not consider any other point's influence.

Linear model = first intuition from human brain -- simplest model
*A model is what?*
-A rule that takes an input and provides an output
-Learns from past examples
-Can identify on unseen data

Linear regression :
$y = $w*$x + $b; $w = weights, $b = bias
---
Note: error function = $(y\^ - y)^2 = ((wx+b)-y)^2$ which is always an upward parabola --> it always has a minimum;
Use gradient descent to find the minimum value of error i.e. \frac{\delta f}{\delta w} & \frac{\delta f}{\delta b};
---
