## Notations:
- $w: l \times m$, where $l$ is the number of code and $m$ is the number of speakers.
- $G: m \times n$ is the channel from speaker array to Lens.
- $A: l \times n$ is the steering vectors towards $l$ directions.
- $\theta: n \times 1$ is the channel of Lens.

## Optimization target:
$$
\max\limits_{\theta, w} ||(wG \cdot A)\theta||_1
$$
$$
s.t. \ \ |\theta_n| = 1\\
|w_l w_l^H| = P_T\\
$$