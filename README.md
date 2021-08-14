# Reverse Engineering Log Curve from 1D LUT

## Running instructions
Install requirements using
```
pip install -r requirements.txt
```

Put a log2linear 1D LUT somewhere, then run:
```
python log_curve.py --lut_file <path to LUT>
```
This will compute the parameters of the log to linear conversion function in about 10 - 15 minutes, depending on your computer's speed.


## Installing the DCTLs in DaVinci Resolve
Go to the `dctl` folder and download the Aces/Zlog conversion DCTLs. On Windows and MacOS, they go in the `IDT` and `ODT` folders located in the following directories:
```
(Windows)
%AppData%\Blackmagic Design\DaVinci Resolve\Support\ACES Transforms\

(MacOS)
~/Library/Application Support/Blackmagic Design/DaVinci Resolve/ACES Transformations/
```

For example, place the aces_to_zlog2.dctl file in the following folder on MacOS:
```
/Users/<your username>/Library/Application Support/Blackmagic Design/DaVinci Resolve/ACES Transformations/ODT/
```
Once you've done that, restart Resolve and they should show up in the ACES Transform node and in the color mangement settings.

The zlog2 to Linear conversion DCTLs should go in Resolve's normal LUTs folder

# Technical Details
Some camera manufacturers do not release the Log to Scene Linear transfer function for their log curves. However, they sometimes provide it in the form of a 1D LUT. The general form of a log to linear conversion is the following function:

```python
def log2linear(x):
    if (t > e * cut + f):
        return (pow(10, (t - d) / c) - b) / a
    else:
        return (t - f) / e
```

And likewise, the inverse of this function is the linear2log curve:
```python
def linear2log(t):
    if (x > cut):
        return c * log10(a * x + b) + d
    else:
        return e * x + f
```

There's only a matter of choosing the correct values for parameters $\theta = \{a, b, c, d, e, f, \text{cut}\}$. As a 1D LUT is constructed by evaluating the ground truth `log2linear` function at $N$ equally spaced points between $0.0$ and $1.0$, this can be viewed as an optimization problem. We simply need to find the correct set of function parameters that result in a minimal error between our constructed `log2linear` function and the ground truth values found in the 1D LUT.

Ultimately, the objective is to derive the `linear2log` conversion, but this is made simpler by matching the `log2linear` curve. This is because the `linear2log` function has a log function in it, limiting its domain to positive numbers, whereas we can safely pass any value into the `log2linear` function and get a differentiable result.

## Optimization
The ground truth `log2linear` function is evaluated in the 1D LUT at $N$ points $X = \{(x_1, y_1), ..., (x_N, y_N)\}$. Our model at iteration $t$ can be described as $f(x, \theta_t)$, and we use the loss function:

$$\ell(\theta_t; X) = \frac{1}{N} \sum_{i = 1}^N \left\lvert ~\log (f(x_i, \theta_t) + 0.5) - \log(y_i + 0.5) \right\rvert$$

Additionally, to find the value of the parameter $\text{cut}$, instead of discretely thresholding when $x > \text{cut}$, I instead use a Sigmoid function with temperature to interpolate from the linear function to the power function. Given a temperature $T$ and $\text{cut}$, we can choose the weight of the power function according to $\sigma((x - \text{cut}) / T)$, allowing both $\text{cut}$ and $T$ to be learnable parameters. If $T$ is small, then we have done a good job of learning the threshold in the piecewise `log2linear` function. The Adam optimizer converged faster than SGD in my tests.
