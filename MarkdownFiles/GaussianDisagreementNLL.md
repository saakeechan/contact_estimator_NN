## Gaussian Disagreement NLL with Teacher Uncertainty

Assume that the teacher and student are noisy estimates of the same underlying true quantity $y^*$:

$$
\begin{aligned}
y_t &= y^* + \epsilon_t, \\
\mu_s &= y^* + \epsilon_s,
\end{aligned}
$$

where the teacher and student errors are modeled as independent Gaussian random variables:

$$
\begin{aligned}
\epsilon_t &\sim \mathcal{N}(0, \sigma_t^2), \\
\epsilon_s &\sim \mathcal{N}(0, \sigma_s^2),
\end{aligned}
$$

and

$$
\operatorname{Cov}(\epsilon_t, \epsilon_s) = 0.
$$

The teacher--student residual is therefore

$$
\begin{aligned}
r &= y_t - \mu_s \\
  &= (y^* + \epsilon_t) - (y^* + \epsilon_s) \\
  &= \epsilon_t - \epsilon_s.
\end{aligned}
$$

Because the two noise sources are independent,

$$
\begin{aligned}
\operatorname{Var}(r) &= \operatorname{Var}(\epsilon_t - \epsilon_s) \\
&= \operatorname{Var}(\epsilon_t) + \operatorname{Var}(\epsilon_s) \\
&= \sigma_t^2 + \sigma_s^2.
\end{aligned}
$$

Hence,

$$
\boxed{r \sim \mathcal{N}\left(0,\, \sigma_t^2 + \sigma_s^2\right)}
$$

and the corresponding Gaussian negative log-likelihood, ignoring the constant $\frac{1}{2}\log(2\pi)$, is

$$
\boxed{
\mathcal{L}_{\mathrm{dis}}
= \frac{1}{2}\frac{(y_t - \mu_s)^2}{\sigma_s^2 + \sigma_t^2}
+ \frac{1}{2}\log\left(\sigma_s^2 + \sigma_t^2\right)
}
$$

### What variance does the student learn?

Let

$$
V = \sigma_s^2 + \sigma_t^2.
$$

Then

$$
\mathcal{L}_{\mathrm{dis}} = \frac{1}{2}\frac{r^2}{V} + \frac{1}{2}\log V.
$$

Differentiating with respect to the student's predicted variance,

$$
\frac{\partial \mathcal{L}_{\mathrm{dis}}}{\partial \sigma_s^2}
= -\frac{r^2}{2(\sigma_s^2 + \sigma_t^2)^2}
+ \frac{1}{2(\sigma_s^2 + \sigma_t^2)}.
$$

Setting the derivative equal to zero,

$$
-\frac{r^2}{(\sigma_s^2 + \sigma_t^2)^2}
+ \frac{1}{\sigma_s^2 + \sigma_t^2} = 0,
$$

which gives

$$
\sigma_s^2 + \sigma_t^2 = r^2,
$$

we obtain

$$
\boxed{\sigma_s^2 = \mathbb{E}[r^2] - \sigma_t^2.}
$$

Thus, the student attempts to learn the portion of the teacher--student disagreement variance that cannot already be explained by the known teacher uncertainty.

### Motivation for the formulation

A standard heteroscedastic Gaussian NLL that ignores teacher uncertainty,

$$
\mathcal{L}_{\mathrm{standard}}
= \frac{1}{2}\frac{(y_t - \mu_s)^2}{\sigma_s^2}
+ \frac{1}{2}\log \sigma_s^2,
$$

implicitly attributes the entire teacher--student residual to the student. Consequently, its learned variance tends toward

$$
\sigma_{s,\mathrm{standard}}^2
\approx \mathbb{E}\left[(y_t - \mu_s)^2\right]
= \sigma_s^2 + \sigma_t^2.
$$

Therefore, uncertainty originating from noisy teacher labels may be incorrectly absorbed into the student's predicted uncertainty.

In contrast, explicitly incorporating the known teacher variance gives

$$
\boxed{\text{Student uncertainty} \approx \text{Residual uncertainty} - \text{Teacher uncertainty}.}
$$

This provides a probabilistic mechanism for separating uncertainty associated with the teacher measurement from uncertainty associated with the student's prediction.

### Advantages

- **Handles noisy labels:** Teacher uncertainty reduces the influence of uncertain residuals.
- **Separates uncertainty sources:** Known teacher variance is not attributed entirely to the student.
- **Well-regularized and extensible:** The log term discourages inflated variance, and the loss generalizes to covariance matrices.

### Limitations and assumptions

- **Independence assumption:** The variance addition $\sigma_s^2 + \sigma_t^2$ is valid only when teacher and student errors are uncorrelated. In general,

  $$
  \operatorname{Var}(\epsilon_t - \epsilon_s)
  = \sigma_t^2 + \sigma_s^2 - 2\operatorname{Cov}(\epsilon_t, \epsilon_s).
  $$

- **Calibrated, independent teacher uncertainty is required:** Correlation, bias, or miscalibration makes the separation unreliable.
- **The single-sample optimum can be negative:** Interpret $\sigma_s^{2*} = r^2 - \sigma_t^2$ across samples, not as literal per-sample subtraction.
- **Gaussian errors are assumed:** Heavy-tailed or multimodal errors are not fully captured.
- **Only the variance sum is identifiable** unless teacher uncertainty is supplied independently.


### Limitation: Teacher and Student Condition on Different Inputs

An additional complication arises when the teacher and student do not condition on the same input information. Let

$$
\mathbf{x}_t = \text{teacher input (e.g., vision/LiDAR)},
\qquad
\mathbf{x}_s = \text{student input (e.g., proprioception)}.
$$

The teacher prediction and uncertainty are conditioned on $\mathbf{x}_t$,

$$
y_t = y^* + \epsilon_t(\mathbf{x}_t),
\qquad
\epsilon_t(\mathbf{x}_t) \sim \mathcal{N}\left(0, \sigma_t^2(\mathbf{x}_t)\right),
$$

whereas the student predicts

$$
\mu_s = f_s(\mathbf{x}_s),
\qquad
\sigma_s^2 = \sigma_s^2(\mathbf{x}_s).
$$

The Gaussian disagreement loss therefore becomes

$$
\mathcal{L}_{\mathrm{dis}}
= \frac{1}{2}
\frac{\left(y_t - \mu_s(\mathbf{x}_s)\right)^2}
{\sigma_s^2(\mathbf{x}_s) + \sigma_t^2(\mathbf{x}_t)}
+ \frac{1}{2}\log\left[\sigma_s^2(\mathbf{x}_s) + \sigma_t^2(\mathbf{x}_t)\right].
$$

Although this likelihood can describe the disagreement between two paired measurements, the interpretation of the learned student variance becomes less straightforward because the two uncertainties are conditioned on different information sets.

In particular,

$$
\sigma_t^2 = \sigma_t^2(\mathbf{x}_t),
\qquad
\sigma_s^2 = \sigma_s^2(\mathbf{x}_s).
$$

Two samples may have nearly identical student inputs,

$$
\mathbf{x}_s^{(1)} \approx \mathbf{x}_s^{(2)},
$$

while having very different teacher inputs and consequently very different teacher uncertainties,

$$
\sigma_t^2(\mathbf{x}_t^{(1)}) \neq \sigma_t^2(\mathbf{x}_t^{(2)}).
$$

The student cannot distinguish these cases using $\mathbf{x}_s$ alone. Consequently, the observed teacher--student residual statistics depend partly on information that is unavailable to the student.

#### Consequence for Variance Subtraction

With independent errors, the disagreement variance is

$$
\operatorname{Var}(y_t - \mu_s) = \sigma_t^2 + \sigma_s^2,
$$

so it is tempting to write

$$
\sigma_s^2 \approx \operatorname{Var}(y_t - \mu_s) - \sigma_t^2.
$$

This is useful for defining the loss, but it is not a literal, per-sample subtraction: the residual depends on both $\mathbf{x}_s$ and $\mathbf{x}_t$, while the student only sees $\mathbf{x}_s$.

#### Interpretation

The disagreement NLL can still train the student from an uncertain teacher:

$$
y_t \sim \mathcal{N}\left(
\mu_s(\mathbf{x}_s),\,
\sigma_s^2(\mathbf{x}_s) + \sigma_t^2(\mathbf{x}_t)
\right),
$$

The learned student variance should be interpreted as uncertainty in the target given the student's own inputs:

$$
\boxed{\sigma_s^2(\mathbf{x}_s) \approx \operatorname{Var}\left(y^* \mid \mathbf{x}_s\right),}
$$
