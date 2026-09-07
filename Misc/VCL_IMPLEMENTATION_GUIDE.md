# Implement Variational Continual Learning for the Bayesian TCN Velocity Estimator

## Objective

Modify the existing Bayes-by-Backprop/UCB-style Bayesian TCN so it implements **Variational Continual Learning (VCL)** as defined by Nguyen et al., *Variational Continual Learning* (ICLR 2018).

The model performs one unchanged regression problem across sequential domains:

- Input: proprioceptive time window (current project preprocessing and shape).
- Output: body-frame velocity, normally `vx, vy, vz`, plus the existing heteroscedastic aleatoric variance output if enabled.
- Task/domain 0: commanded velocity range `0-1 m/s`.
- Task/domain 1: commanded velocity range `1-2 m/s`.
- Later domains may be `2-3 m/s`, new terrain, payload, gait, or other distribution shifts.
- Use one shared output head. Do **not** add task-specific heads or require a task ID at inference.

The first implementation should use explicit dataset/task boundaries for a controlled VCL experiment. Do not add automatic change-point detection yet.

## Mathematical requirement

Represent every Bayesian weight and bias with a diagonal Gaussian posterior

```text
q_t(w_i) = Normal(mu_t[i], sigma_t[i]^2)
sigma_t[i] = softplus(rho_t[i]) + eps
```

At the start of task `t`, freeze a copy of the preceding posterior:

```text
prior_mu[i]    = mu_{t-1}[i]
prior_sigma[i] = sigma_{t-1}[i]
```

Then optimize the negative VCL evidence lower bound:

```text
loss = expected_nll + kl_scale * KL(q_t(w) || q_{t-1}(w))
```

For task 0 only, `q_0`'s prior is the configured initial Gaussian prior, preferably `Normal(0, prior_sigma^2)`. Do not use the fixed zero-mean BBB mixture as the prior after task 0.

For two diagonal Gaussians, implement the analytic elementwise KL:

```text
KL(q || p) = sum_i [
    log(prior_sigma[i] / posterior_sigma[i])
    + (posterior_sigma[i]^2 + (posterior_mu[i] - prior_mu[i])^2)
      / (2 * prior_sigma[i]^2)
    - 0.5
]
```

Clamp standard deviations only for numerical safety, such as `min=1e-8`. The saved prior tensors must be detached and must never receive gradients.

For a minibatch of size `B` drawn from a task dataset containing `N_t` training examples, use one internally consistent estimator. Preferred:

```text
loss = mean_batch_nll + KL_total / N_t
```

This is equivalent to summing likelihoods over the whole task and adding one KL term. Do not divide KL by both the number of minibatches and dataset size. Do not introduce a tunable VCL `lambda` in the faithful baseline; expose an optional diagnostic multiplier defaulting to `1.0` only if useful for ablations.

If the current code defines an epoch objective by summing minibatch losses, `KL_total / M` with `M = number_of_minibatches` is also valid. Do not mix that convention with mean minibatch NLL.

## Regression likelihood

Preserve the current supervised likelihood semantics. For Monte Carlo weight sample `s`:

```text
nll_s = GaussianNLL(target_velocity, predicted_mean_s, predicted_aleatoric_variance_s)
expected_nll = mean_s(nll_s)
```

Continue applying the existing contact mask and final-timestep supervision rules. Reduce the NLL by the number of valid scalar targets, not by the raw batch size when masking changes the valid count.

Use `S` Monte Carlo weight samples during training, initially the existing value `S=10`. The analytic KL is computed once per optimizer step, not once per Monte Carlo sample.

If the repository currently estimates `log q(w) - log p(w)` using sampled weights, replace that estimate with the analytic Gaussian-to-Gaussian KL for the VCL baseline. This reduces Monte Carlo noise and exactly matches the mean-field Gaussian setup.

## Required model changes

### Bayesian layer state

Each Bayesian convolutional/linear layer must contain trainable:

```python
weight_mu
weight_rho
bias_mu       # if bias is enabled
bias_rho
```

It must also store non-trainable prior buffers:

```python
prior_weight_mu
prior_weight_sigma
prior_bias_mu
prior_bias_sigma
```

Register these with `register_buffer` so they:

- move with `.to(device)`;
- appear in `state_dict()`;
- are restored from checkpoints;
- are excluded from optimizer parameters.

Required layer methods:

```python
def posterior_sigma(self): ...
def kl_to_prior(self) -> torch.Tensor: ...
@torch.no_grad()
def set_prior_from_posterior(self): ...
```

`set_prior_from_posterior()` must copy `mu.detach()` and `softplus(rho.detach()) + eps` into the prior buffers. Do not alias storage.

### Model-level API

Add:

```python
def kl_to_prior(self):
    return sum(layer.kl_to_prior() for layer in bayesian_layers)

@torch.no_grad()
def set_prior_from_posterior(self):
    for layer in bayesian_layers:
        layer.set_prior_from_posterior()
```

Do not silently omit Bayesian biases or newly added Bayesian modules. Prefer identifying layers by a common Bayesian-layer base class or protocol, not by fragile parameter-name matching.

## Sequential training procedure

Implement this lifecycle exactly:

```python
initialize_model_with_original_gaussian_prior()

for task_id, task_loader in enumerate(task_loaders):
    if task_id > 0:
        # The prior should already equal the previous task posterior.
        assert_prior_matches_saved_previous_posterior()

    reset_optimizer_for_current_trainable_parameters()

    for epoch in range(num_epochs):
        for batch in task_loader:
            optimizer.zero_grad(set_to_none=True)

            nll = monte_carlo_expected_gaussian_nll(model, batch, S)
            kl = model.kl_to_prior()
            loss = nll + kl / task_dataset_size

            loss.backward()
            optionally_clip_gradients()
            optimizer.step()

    evaluate_on_all_seen_test_sets()
    save_task_posterior_checkpoint(task_id)

    # Commit q_t as the prior for task t+1 only after training and evaluation.
    model.set_prior_from_posterior()
    save_propagation_checkpoint(task_id)
```

Important ordering:

1. Train task `t` against the frozen prior `q_{t-1}`.
2. Evaluate the learned posterior `q_t`.
3. Copy `q_t` into the prior buffers.
4. Train task `t+1`.

Never update the prior buffers during a task, per epoch, or per minibatch. Doing so destroys the VCL constraint by making the reference distribution chase the current posterior.

Reinitialize the optimizer between tasks so Adam momentum from the previous domain does not leak into the next task. Keep posterior parameters; do not reinitialize the model.

## Initialization for task 0

Use a simple Gaussian prior for the faithful baseline:

```text
prior_mu = 0
prior_sigma = configurable scalar, initially 1.0
```

Two acceptable initialization paths are:

1. Initialize posterior means using the current model initialization and posterior standard deviations to a small but trainable value.
2. Pretrain a deterministic maximum-likelihood model on task 0, initialize posterior means from it, and initialize posterior variance small, similar to the paper's experiment setup.

Start with option 1 unless deterministic pretraining already exists. Do not initialize `rho` so negatively that gradients through `softplus(rho)` effectively vanish. Log the initial posterior sigma distribution.

## Inference

There is no task selector. For each input window, sample the single current posterior `q_t(w)` and calculate:

```text
predictive_mean = mean_s(mu_s)
epistemic_variance = variance_s(mu_s)
aleatoric_variance = mean_s(var_s)
total_variance = epistemic_variance + aleatoric_variance
```

Use at least 20-30 Monte Carlo samples for reported uncertainty metrics unless runtime constraints require fewer. Preserve the existing output variance parameterization and floor.

## Checkpoints

Every task checkpoint must contain:

- model `state_dict`, including posterior parameters and prior buffers;
- optimizer state if resuming within the same task;
- completed `task_id`;
- whether prior buffers have already been advanced to the completed posterior;
- VCL scaling convention and task dataset size;
- model/data configuration;
- random seeds.

Avoid ambiguous checkpoints. Prefer two explicit types:

```text
task_01_posterior.pt     # q_1 with prior still q_0; useful for evaluation/audit
task_01_propagation.pt   # q_1 with prior buffers advanced to q_1; ready for task 2
```

On resume, assert that shapes, layer names, and prior lifecycle state match the intended operation.

## Evaluation and logging

After every task, evaluate the same posterior on every seen domain's fixed validation/test split. Produce a matrix:

```text
R[i, j] = error on test domain j after training through domain i
```

At minimum log per domain:

- velocity MAE and RMSE per axis and aggregate;
- Gaussian NLL;
- calibration metric or interval coverage;
- mean aleatoric, epistemic, and total variance;
- `KL_total`, `KL_total / N_t`, NLL, and total loss;
- posterior sigma quantiles per layer;
- mean absolute posterior-mean movement per layer;
- gradient norms for `mu` and `rho`.

Report forgetting for an error metric as:

```text
forgetting_j = error_after_latest_task_on_j - best_previous_error_on_j
```

Also compare against:

1. Naive sequential fine-tuning.
2. Joint training on all domains seen so far (upper-reference baseline, not continual learning).
3. Current fixed-prior BBB/UCB implementation.
4. VCL without a coreset.
5. Optional VCL with a small replay/coreset.

Use identical architecture, data splits, seeds, epochs or optimizer steps, and evaluation code across baselines.

## Tests that must pass

### Unit tests

1. **Analytic KL identity:** KL is approximately zero when posterior and prior means/stds match.
2. **Analytic KL correctness:** compare the implementation against `torch.distributions.kl_divergence` on random diagonal Gaussians.
3. **Gradient isolation:** posterior parameters receive gradients; prior buffers do not.
4. **Prior copy:** after `set_prior_from_posterior()`, every prior buffer numerically equals the detached posterior statistic.
5. **No aliasing:** changing posterior parameters after the copy does not change prior buffers.
6. **Checkpoint round trip:** save/load preserves both posterior and prior exactly.
7. **All Bayesian parameters covered:** KL includes weights and biases from every Bayesian layer.
8. **Loss scaling:** duplicating a dataset without changing its distribution leaves the expected balance consistent under the selected convention.

### Small integration test

Use a tiny synthetic regression stream with two sequential input regions. Verify:

- task 1 starts from task 0 posterior parameters;
- task 1's KL reference remains fixed throughout task 1;
- the prior is advanced only at the boundary;
- evaluation uses one head without a task ID;
- VCL reduces task-0 degradation relative to naive fine-tuning under at least one reproducible configuration.

Do not make the final assertion unrealistically strict: VCL reduces forgetting pressure but does not mathematically guarantee zero forgetting with a mean-field approximation.

## Optional coreset phase - implement only after no-coreset VCL works

The paper's coreset algorithm is more subtle than simply replaying old samples while also propagating a posterior that already contains them; that naive combination double-counts their likelihood.

For this project, first establish no-coreset VCL. Then choose one explicitly labeled extension:

1. **Pragmatic VCL plus replay:** mix a small balanced buffer of old windows into training and acknowledge that this is not exact Algorithm 1.
2. **Faithful coreset VCL:** maintain the paper's non-coreset posterior `q_tilde`, remove held-out coreset points from propagation, and incorporate the current coreset only when forming the prediction posterior.

For an engineering baseline, option 1 is simpler. Sample buffer points across velocity bins, gait/contact regimes, and terrain rather than retaining only the most recent windows. Keep its results separate from pure VCL.

## Explicit non-goals for the first patch

- Do not combine VCL with UCB per-parameter learning-rate scaling.
- Do not retain the original two-component zero-mean mixture prior after task 0.
- Do not add task heads or a task-ID input.
- Do not add epistemic-triggered task discovery yet.
- Do not add replay until pure VCL is verified.
- Do not change the TCN size, dataset preprocessing, masking, or regression target unless required to fix a demonstrated bug.

These exclusions isolate whether posterior propagation itself works.

## Deliverables

1. Bayesian-layer changes implementing frozen previous-posterior priors and analytic KL.
2. Model-level `kl_to_prior()` and `set_prior_from_posterior()` APIs.
3. Sequential VCL trainer with explicit task boundaries.
4. Robust task-boundary checkpoints and resume validation.
5. Unit and synthetic integration tests listed above.
6. Evaluation matrix and forgetting metrics across `0-1 m/s` then `1-2 m/s`.
7. Short README section documenting commands, configs, scaling convention, and checkpoint semantics.

Before editing, inspect the existing Bayesian layer and trainer and report the exact files/functions to modify. Preserve unrelated code and existing user changes. After implementation, run the focused tests and a short smoke-training run; report failures honestly rather than weakening tests.

## Acceptance criteria

The implementation is VCL only if all of the following are true:

- Task `t>0` uses `KL(q_t || q_{t-1})`.
- `q_{t-1}` is a frozen, detached copy of the complete preceding posterior.
- The original prior is used only for task 0.
- Prior buffers stay fixed during each task.
- A single regression head is used at inference.
- KL/NLL scaling is mathematically consistent and logged.
- Checkpoints preserve enough state to resume without corrupting the prior lifecycle.

Reference: Cuong V. Nguyen et al., *Variational Continual Learning*, ICLR 2018, especially equations (1) and (4) and Algorithm 1.
