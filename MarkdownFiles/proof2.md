# Proof: Recovering Student Aleatoric Variance from a Noisy Teacher

## 1. Problem Setup

We have two predictors/sensors estimating the same underlying nominal quantity $y^*$:

- **Student:** observes $x_1$
- **Teacher:** observes $x_2$

The student input $x_1$ represents proprioceptive data. Its noise can arise from sources such as slip and encoder noise.

The teacher input $x_2$ represents visual features from the FAST-LIO/ZED2 perception pipeline. Its noise can arise from the visual features themselves, changing lighting conditions, and related visual-sensing effects.

The student and teacher use two different sensor modalities. Their inputs and noise sources are therefore assumed independent.

The student prediction is modeled as

$$
y_s = \mu_s(x_1) + \epsilon_s,
$$

where

$$
\mathbb{E}[\epsilon_s\mid x_1]=0,
\qquad
\operatorname{Var}(\epsilon_s\mid x_1)=\sigma_s^2(x_1).
$$

The teacher prediction is modeled as

$$
y_t = \mu_t(x_2) + \epsilon_t,
$$

where

$$
\mathbb{E}[\epsilon_t\mid x_2]=0,
\qquad
\operatorname{Var}(\epsilon_t\mid x_2)=\sigma_t^2(x_2).
$$

The teacher provides $\sigma_t^2(x_2)$ for each training sample. The student must learn $\sigma_s^2(x_1)$.

---

## 2. Same Nominal Output Assumption

Assume that, for the corresponding physical state, the teacher and student estimate the same nominal output:

$$
\mu_s(x_1)=\mu_t(x_2)=y^*.
$$

Therefore,

$$
y_s = y^*+\epsilon_s,
$$

$$
y_t = y^*+\epsilon_t.
$$

This assumption removes systematic mean disagreement between the teacher and student. Their disagreement is therefore due only to their respective noise terms.

---

## 3. Define the Teacher–Student Residual

Define

$$
r = y_t-y_s.
$$

Substituting the two observation models gives

$$
r
= (y^*+\epsilon_t)-(y^*+\epsilon_s),
$$

and hence

$$
\boxed{r=\epsilon_t-\epsilon_s.}
$$

The common nominal output cancels.

---

## 4. Mean of the Residual

Because both noise terms are zero mean,

$$
\mathbb{E}[r\mid x_1,x_2]
=
\mathbb{E}[\epsilon_t-\epsilon_s\mid x_1,x_2].
$$

Under the assumed noise model,

$$
\boxed{\mathbb{E}[r\mid x_1,x_2]=0.}
$$

Thus the residual contains no systematic bias under the same-nominal-output assumption.

---

## 5. Variance of the Residual

From

$$
r=\epsilon_t-\epsilon_s,
$$

we obtain

$$
\operatorname{Var}(r\mid x_1,x_2)
=
\operatorname{Var}(\epsilon_t-\epsilon_s\mid x_1,x_2).
$$

In general,

$$
\operatorname{Var}(A-B)
=
\operatorname{Var}(A)
+
\operatorname{Var}(B)
-
2\operatorname{Cov}(A,B).
$$

Therefore,

$$
\operatorname{Var}(r\mid x_1,x_2)
=
\sigma_t^2(x_2)
+
\sigma_s^2(x_1)
-
2\operatorname{Cov}(\epsilon_t,\epsilon_s\mid x_1,x_2).
$$

Because the teacher and student have independent noise sources,

$$
\operatorname{Cov}(\epsilon_t,\epsilon_s\mid x_1,x_2)=0.
$$

Hence

$$
\boxed{
\operatorname{Var}(r\mid x_1,x_2)
=
\sigma_s^2(x_1)+\sigma_t^2(x_2).
}
$$

This is the central variance-addition result.

---

## 6. Recovering the Student Variance

Rearranging the previous equation gives

$$
\boxed{
\sigma_s^2(x_1)
=
\operatorname{Var}(r\mid x_1,x_2)
-
\sigma_t^2(x_2).
}
$$

Therefore, conceptually,

$$
\boxed{
\text{student aleatoric variance}
=
\text{teacher–student residual variance}
-
\text{teacher aleatoric variance}.
}
$$

---



## 7. The Training Likelihood

Instead of explicitly estimating residual variance and subtracting teacher variance, model the residual as

$$
r\mid x_1,x_2
\sim
\mathcal N\left(
0,
\sigma_s^2(x_1)+\sigma_t^2(x_2)
\right).
$$

The corresponding negative log-likelihood, ignoring constants independent of the learned parameters, is

$$
\boxed{
\mathcal L
=
\frac{1}{2}
\left[
\frac{r^2}
{\sigma_s^2(x_1)+\sigma_t^2(x_2)}
+
\log\left(
\sigma_s^2(x_1)+\sigma_t^2(x_2)
\right)
\right].
}
$$

The teacher variance $\sigma_t^2(x_2)$ is supplied for each sample. The network predicts only $\sigma_s^2(x_1)$.

---

## 8. What Does the Loss Learn?

For a fixed student input $x_1$, define

$$
v=\sigma_s^2(x_1)
$$

and

$$
t=\sigma_t^2(x_2).
$$

The expected population loss is

$$
J(v\mid x_1)
=
\mathbb{E}
\left[
\frac{1}{2}
\left(
\frac{r^2}{v+t}
+
\log(v+t)
\right)
\middle|x_1
\right].
$$

Differentiate with respect to $v$:

$$
\frac{\partial J}{\partial v}
=
\frac{1}{2}
\mathbb{E}
\left[
-\frac{r^2}{(v+t)^2}
+\frac{1}{v+t}
\middle|x_1
\right].
$$

Combining terms,

$$
\boxed{
\frac{\partial J}{\partial v}
=
\frac{1}{2}
\mathbb{E}
\left[
\frac{v+t-r^2}{(v+t)^2}
\middle|x_1
\right].
}
$$

At the population optimum,

$$
\mathbb{E}
\left[
\frac{v+t-r^2}{(v+t)^2}
\middle|x_1
\right]=0.
$$

---

## 9. Show That the True Student Variance Is an Optimum

Let the true student aleatoric variance be

$$
v^*=\sigma_{s,\mathrm{true}}^2(x_1).
$$

From the generative model,

$$
\mathbb{E}[r^2\mid x_1,x_2]
=
v^*+t.
$$

Evaluate the expected gradient at $v=v^*$:

$$
\left.\frac{\partial J}{\partial v}\right|_{v=v^*}
=
\frac12
\mathbb{E}
\left[
\frac{v^*+t-r^2}{(v^*+t)^2}
\middle|x_1
\right].
$$

By the law of iterated expectations, we may condition first on $x_2$:

$$
\mathbb{E}
\left[
\frac{v^*+t-r^2}{(v^*+t)^2}
\middle|x_1
\right]
=
\mathbb{E}_{x_2\mid x_1}
\left[
\mathbb{E}
\left[
\frac{v^*+t-r^2}{(v^*+t)^2}
\middle|x_1,x_2
\right]
\right].
$$

For fixed $(x_1,x_2)$, the denominator is fixed, so

$$
\mathbb{E}
\left[
\frac{v^*+t-r^2}{(v^*+t)^2}
\middle|x_1,x_2
\right]
=
\frac{
v^*+t-\mathbb{E}[r^2\mid x_1,x_2]
}{(v^*+t)^2}.
$$

Using

$$
\mathbb{E}[r^2\mid x_1,x_2]=v^*+t,
$$

we obtain

$$
\frac{v^*+t-(v^*+t)}{(v^*+t)^2}=0.
$$

Therefore,

$$
\boxed{
\left.\frac{\partial J}{\partial v}\right|_{v=v^*}=0.
}
$$

Thus the true student aleatoric variance is a stationary population solution of the correctly specified Gaussian likelihood.

Under the usual identifiability, sufficient-data, optimization, and model-capacity assumptions, maximum-likelihood training is therefore consistent with learning

$$
\boxed{
\sigma_s^2(x_1)\rightarrow\sigma_{s,\mathrm{true}}^2(x_1).
}
$$

---

## 10. What Happens to the Varying Teacher Variance?

The teacher variance may vary strongly with $x_2$:

$$
\sigma_t^2=0.001,\;0.1,\;0.03,\;1.2,\ldots
$$

The student does **not** need to predict $x_2$, learn $\sigma_t^2(x_2)$, or manually subtract an average teacher variance before training.

For every sample, the supplied teacher variance appears directly in

$$
\sigma_s^2(x_1)+\sigma_t^2(x_2).
$$

Consequently, each residual is interpreted relative to the amount of teacher noise present in that particular sample.

Across the training distribution, optimization performs the required averaging/marginalization implicitly.

---

## 11. Relation to the Marginal Variance Equation

If one additionally assumes

$$
x_1\perp x_2,
$$

then

$$
\mathbb{E}[\sigma_t^2(x_2)\mid x_1]
=
\mathbb{E}[\sigma_t^2(x_2)].
$$

Marginalizing the residual second moment over $x_2$ gives

$$
\mathbb{E}[r^2\mid x_1]
=
\sigma_s^2(x_1)
+
\mathbb{E}[\sigma_t^2(x_2)].
$$

Hence

$$
\boxed{
\sigma_s^2(x_1)
=
\mathbb{E}[r^2\mid x_1]
-
\mathbb{E}[\sigma_t^2(x_2)].
}
$$

This equation describes the **marginal moment relationship**. It is not necessary to explicitly compute this subtraction when training with the sample-wise Gaussian likelihood above.

---

## 12. Assumptions Required for the Proof

The result depends on the following assumptions:

1. **Same nominal output**

   $$
   \mu_s(x_1)=\mu_t(x_2)=y^*.
   $$

2. **Zero-mean student noise**

   $$
   \mathbb{E}[\epsilon_s\mid x_1]=0.
   $$

3. **Zero-mean teacher noise**

   $$
   \mathbb{E}[\epsilon_t\mid x_2]=0.
   $$

4. **Independent student and teacher noise sources**

   Because $x_1$ and $x_2$ come from two different sensor modalities (proprioceptive and visual), their noise sources are assumed independent.

   $$
   \operatorname{Cov}(\epsilon_s,\epsilon_t\mid x_1,x_2)=0.
   $$

5. **Teacher covariance is calibrated and known for each training sample**

   $$
   \sigma_t^2(x_2)
   =
   \operatorname{Var}(\epsilon_t\mid x_2).
   $$

6. **The likelihood model is correctly specified**, or sufficiently close to the actual residual distribution for the intended interpretation.

7. **Independent student and teacher inputs for the additional global-average simplification in Section 12 only**

   Because $x_1$ and $x_2$ arise from different sensor modalities, assume

   $$
   x_1\perp x_2.
   $$

The independence of $x_1$ and $x_2$ is **not required** for the basic variance-addition equation or for the sample-wise likelihood argument. Independence of the **noise terms** is the key requirement for variance addition.

---

## 13. Final Result

Under these assumptions,

$$
\boxed{
\operatorname{Var}(y_t-y_s\mid x_1,x_2)
=
\sigma_s^2(x_1)+\sigma_t^2(x_2)
}
$$

and therefore

$$
\boxed{
\sigma_s^2(x_1)
=
\operatorname{Var}(y_t-y_s\mid x_1,x_2)
-
\sigma_t^2(x_2).
}
$$

Training with

$$
\boxed{
\mathcal L
=
\frac12
\left[
\frac{(y_t-y_s)^2}
{\sigma_s^2(x_1)+\sigma_t^2(x_2)}
+
\log\left(\sigma_s^2(x_1)+\sigma_t^2(x_2)\right)
\right]
}
$$

allows the student to recover its own aleatoric variance without explicitly estimating or subtracting the teacher's average variance, provided the model assumptions hold.
