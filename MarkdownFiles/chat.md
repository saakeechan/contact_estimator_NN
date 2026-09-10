from pathlib import Path

md = r"""# Codex Implementation Spec — UCB Continual Learning for Velocity Estimation

## Goal

Implement **Uncertainty-guided Continual Learning with Bayesian Neural Networks (UCB)** for a regression model that predicts velocity.

The continual-learning experiment is sequential:

- **Task 1:** command velocity in `[0, 1] m/s`
- **Task 2:** command velocity in `[2, 3] m/s`

The model must learn Task 2 while retaining performance on Task 1.

Do **not** implement VCL. In particular, do not use the previous posterior as the prior for the next task.

UCB instead:

1. Trains a Bayesian neural network using **Bayes by Backprop (BBB)**.
2. Uses the learned posterior uncertainty of each weight to modify that weight's learning rate before the next task.

---

# 1. Bayesian parameterization

Every trainable scalar weight should have a Gaussian variational posterior:

\[
q(w_i) = \mathcal N(\mu_i,\sigma_i^2)
\]

Parameterize the standard deviation with an unconstrained parameter `rho`:

\[
\sigma_i = \operatorname{softplus}(\rho_i)
= \log(1+\exp(\rho_i))
\]

During training, sample weights using the reparameterization trick:

\[
\epsilon_i \sim \mathcal N(0,1)
\]

\[
w_i = \mu_i + \sigma_i \epsilon_i
\]

Implementation requirement:

```python
sigma = F.softplus(rho)
eps = torch.randn_like(mu)
weight = mu + sigma * eps

Create Bayesian equivalents of the required deterministic layers, e.g.

BayesianLinear
BayesianConv1d

Each Bayesian layer should store:

weight_mu
weight_rho
bias_mu
bias_rho

and should expose the sampled weight/bias used in the forward pass.

2. Prior

Use the scale-mixture Gaussian prior used in Bayes by Backprop / UCB:

$$ p(w) = \pi \mathcal N(0,\sigma_1^2) + (1-\pi)\mathcal N(0,\sigma_2^2) $$

Make these configurable:

prior_pi
prior_sigma1
prior_sigma2

Do not silently replace this with:

N(0, 1)

unless explicitly enabled as an ablation.

Compute the prior log probability stably using log-sum-exp.

For one weight tensor:

log_prob_1 = Normal(0, sigma1).log_prob(w) + log(pi)
log_prob_2 = Normal(0, sigma2).log_prob(w) + log(1 - pi)

log_prior = torch.logsumexp(
    torch.stack([log_prob_1, log_prob_2]),
    dim=0
).sum()
3. Posterior log probability

For each sampled weight tensor:

$$ \log q(w|\mu,\sigma) = \sum_i \log \mathcal N(w_i|\mu_i,\sigma_i^2) $$

Each Bayesian layer should accumulate:

log_prior
log_variational_posterior

for the current sampled forward pass.

After the model forward pass, sum these quantities across all Bayesian layers.

4. Regression likelihood

This project is regression, not classification.

The data term should be a regression negative log likelihood.

If the network currently predicts only a mean velocity:

$$ \hat y = f_w(x) $$

start with a fixed observation-noise model:

$$ p(y|x,w) = \mathcal N(y;f_w(x),\sigma_y^2) $$

Then

$$ -\log p(D|w) $$

is Gaussian NLL.

If the existing model already predicts aleatoric variance, preserve the existing heteroscedastic Gaussian NLL:

$$ \mathcal L_{\mathrm{NLL}} = \frac12 \left[ \frac{(y-\mu_y)^2}{\sigma_y^2} + \log \sigma_y^2 \right] $$

Do not confuse:

weight posterior variance \(\sigma_w^2\): used by UCB for continual learning
output aleatoric variance \(\sigma_y^2\): observation uncertainty

They are different quantities.

5. BBB objective

For each minibatch, optimize the negative ELBO / Bayes-by-Backprop objective.

Conceptually:

$$ \mathcal L_{\mathrm{BBB}} = \frac{1}{M} \left( \log q(w) - \log p(w) \right) - \log p(D|w) $$

where M = number of minibatches in one epoch.

Equivalent implementation:

complexity_cost = (log_q - log_prior) / num_batches
data_loss = regression_nll(prediction, target)

loss = complexity_cost + data_loss

Be careful with scaling.

The UCB pseudocode writes:

$$ \frac{1}{M}(l_1-l_2-l_3) $$

but in minibatch implementations the important requirement is that the KL / complexity term is correctly scaled relative to the minibatch likelihood.

Keep the scaling explicit and configurable.

6. Monte Carlo samples during training

Support:

num_mc_train_samples

Default initially:

num_mc_train_samples = 1

For every Monte Carlo sample:

sample new Bayesian weights
forward pass
calculate log posterior
calculate log prior
calculate regression likelihood

Average the resulting loss over MC samples.

Pseudo-code:

loss = 0

for _ in range(num_mc_train_samples):
    pred = model(x, sample=True)

    log_q = model.log_variational_posterior()
    log_prior = model.log_prior()

    nll = regression_nll(pred, y)

    loss += (log_q - log_prior) / num_batches + nll

loss /= num_mc_train_samples
7. Optimizer architecture

UCB requires different effective learning rates for different Bayesian parameters.

A normal optimizer with one scalar learning rate is insufficient if we want to reproduce the algorithm exactly.

Maintain element-wise learning-rate multipliers for the posterior means.

For every Bayesian parameter tensor, maintain:

lr_multiplier_mu

with the same shape as mu.

Initially:

$$ \alpha_{\mu,i} = \alpha_0 $$

for every parameter.

For rho, standard UCB uses:

$$ \Omega_\rho = 1 $$

so its learning rate remains unchanged.

8. Gradient update for UCB

For a posterior mean parameter:

$$ \mu_i \leftarrow \mu_i - \alpha_{\mu,i} \frac{\partial L}{\partial \mu_i} $$

Implement this either by:

Option A — gradient scaling

Before optimizer.step():

param_mu.grad *= lr_multiplier_mu

with the base optimizer learning rate treated as alpha0.

or

Option B — manual SGD update
mu -= elementwise_lr_mu * mu.grad

Option A is preferable if the existing training stack uses Adam, but document clearly that this is then an adaptation of the exact SGD-style pseudocode.

For the cleanest reproduction of UCB, first implement with SGD.

9. UCB learning-rate update after each task

After Task t is completely trained, compute:

$$ \sigma_i = \operatorname{softplus}(\rho_i) $$

The UCB paper defines:

$$ \Omega_{\mu,i} = \frac{1}{\sigma_i} $$

and then:

$$ \alpha_{\mu,i} \leftarrow \frac{\alpha_{\mu,i}}{\Omega_{\mu,i}} $$

Therefore:

$$ \boxed{ \alpha_{\mu,i} \leftarrow \alpha_{\mu,i}\sigma_i } $$

This is the central continual-learning operation.

Interpretation:

small posterior sigma
    -> high confidence in weight
    -> small future learning rate
    -> preserve weight

large posterior sigma
    -> low confidence in weight
    -> larger future learning rate
    -> allow adaptation

For rho:

$$ \Omega_{\rho,i}=1 $$

so:

$$ \alpha_{\rho,i}\leftarrow\alpha_{\rho,i} $$

i.e. unchanged in standard UCB.

Implementation:

@torch.no_grad()
def update_ucb_learning_rates(model):
    for layer in bayesian_layers(model):
        sigma_w = F.softplus(layer.weight_rho)

        layer.weight_lr_multiplier *= sigma_w

        if layer.bias_mu is not None:
            sigma_b = F.softplus(layer.bias_rho)
            layer.bias_lr_multiplier *= sigma_b

IMPORTANT:

Repeated tasks make this update multiplicative:

$$ \alpha_i^{(t+1)} = \alpha_i^{(t)}\sigma_i^{(t)} $$

Do not reset all learning rates to alpha0 at the start of a new task.

10. Numerical stabilization

Posterior standard deviations may produce pathological learning-rate multipliers.

Add configurable safeguards:

sigma_min
sigma_max
lr_multiplier_min
lr_multiplier_max

For example:

sigma_for_ucb = sigma.clamp(min=sigma_min, max=sigma_max)
lr_multiplier *= sigma_for_ucb
lr_multiplier.clamp_(lr_multiplier_min, lr_multiplier_max)

However:

keep an exact_ucb=True mode with no clipping
use clipping only as an explicit engineering stabilization option
log how often clipping occurs

Do not change the algorithm silently.

11. Task construction

Create two separate datasets.

Task 1
command velocity: 0 <= v_cmd <= 1 m/s

Create:

task1_train
task1_val
task1_test
Task 2
command velocity: 2 <= v_cmd <= 3 m/s

Create:

task2_train
task2_val
task2_test

No Task-1 training samples should be replayed while training Task 2 for the primary UCB experiment.

That is important because the experiment is testing whether UCB itself reduces catastrophic forgetting.

12. Sequential training protocol

Implement:

initialize Bayesian model

TRAIN TASK 1
    train only on task1_train
    validate on task1_val
    stop when converged

EVALUATE AFTER TASK 1
    evaluate task1_test
    store metrics as T1_after_T1

UPDATE UCB LEARNING RATES
    posterior sigma -> element-wise learning-rate multipliers

TRAIN TASK 2
    train only on task2_train
    validate on task2_val
    do NOT replay task1_train

EVALUATE AFTER TASK 2
    evaluate task1_test
    store metrics as T1_after_T2

    evaluate task2_test
    store metrics as T2_after_T2
13. Required evaluation metrics

At minimum record regression:

MAE
RMSE

for every task after every training stage.

Important quantities:

$$ E_{1,1} = \text{Task-1 error immediately after Task 1} $$ $$ E_{1,2} = \text{Task-1 error after learning Task 2} $$ $$ E_{2,2} = \text{Task-2 error after learning Task 2} $$

Define forgetting for an error metric as:

$$ F_1 = E_{1,2}-E_{1,1} $$

Interpretation:

F1 ~= 0     -> little forgetting
F1 > 0      -> performance deteriorated
large F1    -> catastrophic forgetting

Also record Task-2 learning performance using E_2,2.

14. Required baselines

Implement these as separate experiment modes.

Baseline A — ordinary sequential fine-tuning

Train:

Task 1 -> Task 2

using the same architecture/training setup but without UCB learning-rate protection.

This is the most important comparison.

Baseline B — joint training

Train on:

Task 1 + Task 2

simultaneously.

This is an approximate upper-bound/reference and is not continual learning.

Method C — UCB

Train sequentially using the Bayesian model and uncertainty-guided learning rates.

Comparison table should ultimately contain:

Method	T1 after T1	T1 after T2	T2 after T2	Forgetting
Fine-tuning				
Joint				
UCB				
15. Bayesian inference at test time

Support two modes.

Posterior mean inference

Use:

$$ w=\mu $$

No sampling.

prediction = model(x, sample=False)

Use this for the main deterministic regression accuracy comparison unless there is a reason to use Bayesian model averaging.

Monte Carlo Bayesian prediction

Sample weights K times:

$$ w^{(k)}\sim q(w) $$

and compute:

$$ \bar y = \frac1K\sum_k f_{w^{(k)}}(x) $$

Also calculate predictive epistemic variance:

$$ \sigma_{\mathrm{epi}}^2 = \frac1K \sum_k (f_{w^{(k)}}(x)-\bar y)^2 $$

Expose:

predict_mc(x, num_samples=K)

returning:

predictive_mean
epistemic_variance
16. Logging required for debugging UCB

After every task, log distributions/statistics of:

posterior sigma
mu
|mu| / sigma
element-wise mu learning-rate multiplier

At minimum report:

mean
std
min
median
95th percentile
max

Also log per layer.

This is essential to check whether UCB is actually creating:

low-sigma / protected weights
high-sigma / plastic weights

rather than collapsing all learning rates.

17. Sanity checks

Before running the full experiment, implement these tests.

Test 1 — sampling

For a Bayesian scalar parameter with fixed mu and rho, sample many times and verify empirically:

sample mean ~= mu
sample std ~= softplus(rho)
Test 2 — posterior log probability

Compare custom log q(w) against:

torch.distributions.Normal(mu, sigma).log_prob(w)
Test 3 — mixture prior

Verify the mixture prior log probability against a direct numerical implementation.

Test 4 — UCB update direction

Create:

sigma_A = 0.1
sigma_B = 0.8
same starting learning rate

After UCB update verify:

lr_A < lr_B

Specifically:

$$ \alpha_A^{new}=0.1\alpha_A^{old} $$ $$ \alpha_B^{new}=0.8\alpha_B^{old} $$
Test 5 — no accidental replay

While training Task 2, verify that no Task-1 samples enter the training dataloader.

Test 6 — no LR reset

Verify that the Task-2 mean learning-rate multipliers are exactly those produced at the end of Task 1.

18. Suggested code organization
models/
    bayesian_layers.py
    bayesian_velocity_model.py

losses/
    bbb.py
    regression_nll.py

continual/
    ucb.py

training/
    train_task.py
    evaluate.py

data/
    build_velocity_tasks.py

experiments/
    run_ucb_velocity.py

tests/
    test_bayesian_layers.py
    test_bbb_loss.py
    test_ucb_update.py
19. Suggested APIs
class BayesianLinear(nn.Module):
    def forward(self, x, sample=True):
        ...

    def log_prior(self):
        ...

    def log_variational_posterior(self):
        ...
class BayesianVelocityModel(nn.Module):
    def forward(self, x, sample=True):
        ...

    def log_prior(self):
        ...

    def log_variational_posterior(self):
        ...

    def posterior_sigmas(self):
        ...
def bbb_loss(
    model,
    x,
    y,
    num_batches,
    num_mc_samples=1,
):
    ...
def apply_ucb_lr_update(model):
    ...
def train_task(
    model,
    train_loader,
    val_loader,
    config,
):
    ...
def evaluate_regression(
    model,
    loader,
):
    ...
20. Important conceptual constraints

Do not implement any of the following unless requested as a separate ablation:

previous posterior as next-task prior
replay buffer
experience replay
EWC
LwF
teacher-student distillation
parameter freezing based on Fisher information

Those are different continual-learning mechanisms.

The target method is:

$$ \boxed{ \text{Bayes by Backprop} \rightarrow \text{posterior weight uncertainty} \rightarrow \text{per-weight learning rates} \rightarrow \text{sequential training} } $$
21. First implementation milestone

Do not begin with the entire existing robotics model.

First prove the machinery works with a very small Bayesian MLP on the same Task-1 / Task-2 data split.

Milestone:

Bayesian MLP
-> Task 1 train
-> measure sigma
-> apply UCB LR update
-> Task 2 train
-> evaluate T1 forgetting and T2 learning

Only after this works should the Bayesian/UCB machinery be inserted into the full velocity-estimation architecture.

Reference algorithm

Primary method:

Ebrahimi et al., "Uncertainty-guided Continual Learning with Bayesian Neural Networks," ICLR 2020.

UCB uses Bayes-by-Backprop for Bayesian weight learning, then updates parameter learning rates according to posterior weight uncertainty.

Bayes-by-Backprop background:

Blundell et al., "Weight Uncertainty in Neural Networks," ICML 2015.
"""

path = Path("/mnt/data/ucb_continual_learning_codex_spec.md")
path.write_text(md)
print(f"Created: {path}")
print(f"{len(md.splitlines())} lines")