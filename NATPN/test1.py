"""Train the bundled Natural Posterior Network on a noisy 3D regression toy problem."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


NATPN_ROOT = Path(__file__).with_name("natural-posterior-network")
if not NATPN_ROOT.is_dir():
    raise RuntimeError(f"Bundled NatPN source is missing: {NATPN_ROOT}")
sys.path.insert(0, str(NATPN_ROOT))

from natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel  # noqa: E402
from natpn.nn.encoder import TabularEncoder  # noqa: E402
from natpn.nn.flow import RadialFlow  # noqa: E402
from natpn.nn.output import NormalOutput  # noqa: E402


SAVE_PLOTS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    x_min: float = -2.0
    x_max: float = 2.0
    train_samples: int = 8_000
    val_samples: int = 2_000
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    entropy_weight: float = 1e-5
    warmup_epochs: int = 3
    run_finetuning: bool = True
    latent_dim: int = 16
    flow_layers: int = 8
    certainty_budget: str = "normal"
    seed: int = 0
    test_x1: float = 1.0
    test_x2: float = 2.0
    test_x3: float = 2.0
    checkpoint_path: str = "natpn_test1.pt"
    loss_plot_path: str = "natpn_test1_loss.png"
    umap_plot_path: str = "natpn_test1_input_umap.png"


def true_mean(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return torch.sin(1.5 * x1) + 0.4 * x2.square() - 0.3 * x1 * x2 + 0.5 * torch.cos(x3)


def true_variance(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.08 * (x1.square() + torch.sin(x2).square() + 0.5 * x3.square())


def make_dataset(num_samples: int, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty(num_samples, 3, device=DEVICE).uniform_(cfg.x_min, cfg.x_max)
    mean = true_mean(x[:, 0], x[:, 1], x[:, 2])
    variance = true_variance(x[:, 0], x[:, 1], x[:, 2])
    return x, mean + variance.sqrt() * torch.randn_like(mean)


def make_model(cfg: Config) -> NaturalPosteriorNetworkModel:
    return NaturalPosteriorNetworkModel(
        latent_dim=cfg.latent_dim,
        encoder=TabularEncoder(3, [64, 64], cfg.latent_dim),
        flow=RadialFlow(cfg.latent_dim, cfg.flow_layers),
        output=NormalOutput(cfg.latent_dim),
        certainty_budget=cfg.certainty_budget,  # type: ignore[arg-type]
    ).to(DEVICE)


def posterior_variances(posterior: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aleatoric, epistemic, and predictive variances from NatPN's Normal-Gamma posterior."""
    alpha = posterior.alpha  # type: ignore[attr-defined]
    beta = posterior.beta  # type: ignore[attr-defined]
    lambd = posterior.lambd  # type: ignore[attr-defined]
    aleatoric = beta / (alpha - 1.0).clamp_min(1e-6)
    epistemic = aleatoric / lambd
    return aleatoric, epistemic, aleatoric + epistemic


def likelihood_aleatoric_variance(model: NaturalPosteriorNetworkModel, x: torch.Tensor) -> torch.Tensor:
    """Observation variance from the output head, before NatPN's prior update."""
    likelihood = model.output(model.encoder(x))
    return likelihood.precision.reciprocal()


def run_epoch(
    model: NaturalPosteriorNetworkModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: BayesianLoss,
    *,
    flow_only: bool = False,
) -> tuple[float, float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_sse = total_alea = total_epi = total_log_prob = 0.0
    total_count = 0

    for x, y in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        if flow_only:
            log_prob = model.log_prob(x, track_encoder_gradients=False)
            loss = -log_prob.mean()
            prediction = None
        else:
            prediction, log_prob = model(x)
            loss = loss_fn(prediction, y)

        if training:
            loss.backward()
            optimizer.step()

        count = y.numel()
        total_loss += loss.item() * count
        total_log_prob += log_prob.detach().sum().item()
        if prediction is not None:
            mean = prediction.maximum_a_posteriori().mean()
            aleatoric, epistemic, _ = posterior_variances(prediction)
            total_sse += (mean.detach() - y).square().sum().item()
            total_alea += aleatoric.detach().sum().item()
            total_epi += epistemic.detach().sum().item()
        total_count += count

    if flow_only:
        return total_loss / total_count, float("nan"), float("nan"), float("nan"), total_log_prob / total_count
    return (
        total_loss / total_count,
        (total_sse / total_count) ** 0.5,
        total_alea / total_count,
        total_epi / total_count,
        total_log_prob / total_count,
    )


def plot_loss_history(train_losses: list[float], val_losses: list[float], cfg: Config) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train Bayesian loss")
    ax.plot(epochs, val_losses, label="validation Bayesian loss")
    ax.set(xlabel="epoch", ylabel="loss", title="NatPN Training and Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.loss_plot_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def plot_input_umap(model: NaturalPosteriorNetworkModel, train_x: torch.Tensor, cfg: Config) -> None:
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(random_state=cfg.seed)
    embedding = reducer.fit_transform(train_x.cpu().numpy())
    test_embedding = reducer.transform([[cfg.test_x1, cfg.test_x2, cfg.test_x3]])
    log_prob = model.log_prob(train_x).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(embedding[:, 0], embedding[:, 1], c=log_prob, s=8, alpha=0.65, cmap="viridis")
    ax.scatter(*test_embedding[0], marker="*", s=220, c="crimson", edgecolors="black", linewidths=0.8)
    ax.set(title="UMAP of 3D inputs", xlabel="UMAP 1", ylabel="UMAP 2")
    fig.colorbar(points, ax=ax, label="NatPN latent log density")
    fig.tight_layout()
    fig.savefig(cfg.umap_plot_path, dpi=200)
    plt.close(fig)


def optimize_flow(
    model: NaturalPosteriorNetworkModel,
    loader: DataLoader,
    cfg: Config,
    epochs: int,
    label: str,
) -> None:
    optimizer = torch.optim.Adam(model.flow.parameters(), lr=cfg.learning_rate)
    for epoch in range(1, epochs + 1):
        loss, _, _, _, log_prob = run_epoch(model, loader, optimizer, BayesianLoss(), flow_only=True)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{label} {epoch:03d} | flow_nll={loss:.4f} | log_prob={log_prob:.4f}")


def train(model: NaturalPosteriorNetworkModel, cfg: Config) -> None:
    train_x, train_y = make_dataset(cfg.train_samples, cfg)
    val_x, val_y = make_dataset(cfg.val_samples, cfg)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=cfg.batch_size)
    loss_fn = BayesianLoss(cfg.entropy_weight)

    if cfg.warmup_epochs:
        optimize_flow(model, train_loader, cfg, cfg.warmup_epochs, "warmup")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_losses, val_losses = [], []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_rmse, train_alea, train_epi, train_log_prob = run_epoch(model, train_loader, optimizer, loss_fn)
        with torch.no_grad():
            val_loss, val_rmse, val_alea, val_epi, val_log_prob = run_epoch(model, val_loader, None, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_rmse={train_rmse:.4f} | val_rmse={val_rmse:.4f} | "
                f"train_alea={train_alea:.4f} | val_alea={val_alea:.4f} | "
                f"train_epi={train_epi:.4f} | val_epi={val_epi:.4f} | "
                f"train_log_prob={train_log_prob:.4f} | val_log_prob={val_log_prob:.4f}"
            )

    if cfg.run_finetuning:
        optimize_flow(model, train_loader, cfg, cfg.epochs, "fine-tune")

    if SAVE_PLOTS:
        plot_loss_history(train_losses, val_losses, cfg)
        plot_input_umap(model, train_x, cfg)
        print(f"Saved {Path(cfg.loss_plot_path).resolve()}")
        print(f"Saved {Path(cfg.umap_plot_path).resolve()}")

    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"Saved checkpoint to {Path(cfg.checkpoint_path).resolve()}")


@torch.no_grad()
def test(model: NaturalPosteriorNetworkModel, cfg: Config) -> None:
    test_x, test_y = make_dataset(cfg.val_samples, cfg)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg.batch_size)
    loss, rmse, aleatoric, epistemic, log_prob = run_epoch(model, test_loader, None, BayesianLoss(cfg.entropy_weight))
    print(
        f"test_loss={loss:.4f} | test_rmse={rmse:.4f} | mean_aleatoric_var={aleatoric:.4f} | "
        f"mean_epistemic_var={epistemic:.4f} | mean_log_prob={log_prob:.4f}"
    )

    x = torch.tensor([[cfg.test_x1, cfg.test_x2, cfg.test_x3]], device=DEVICE)
    raw_aleatoric = likelihood_aleatoric_variance(model, x)
    posterior, log_prob = model(x)
    predicted_mean = posterior.maximum_a_posteriori().mean()
    aleatoric, epistemic, total = posterior_variances(posterior)
    print(
        f"specified input: x1={cfg.test_x1:.4f} | x2={cfg.test_x2:.4f} | x3={cfg.test_x3:.4f} | "
        f"predicted_y={predicted_mean.item():.4f} | raw_aleatoric_var={raw_aleatoric.item():.4f} | "
        f"aleatoric_var={aleatoric.item():.4f} | "
        f"epistemic_var={epistemic.item():.4f} | predictive_var={total.item():.4f} | "
        f"latent_log_prob={log_prob.item():.4f} | "
        f"true_y_mean={true_mean(x[:, 0], x[:, 1], x[:, 2]).item():.4f} | "
        f"true_variance={true_variance(x[:, 0], x[:, 1], x[:, 2]).item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-train", action="store_true", help="train, save a checkpoint, then test")
    mode.add_argument("-test", action="store_true", help="load the checkpoint and test (the default)")
    args = parser.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    print(
        f"Using {DEVICE}; NatPN regression with a {cfg.latent_dim}D radial-flow latent space "
        f"({cfg.flow_layers} transforms, {cfg.certainty_budget} certainty budget)."
    )
    model = make_model(cfg)
    if args.train:
        train(model, cfg)
    else:
        checkpoint_path = Path(cfg.checkpoint_path)
        if not checkpoint_path.is_file():
            parser.error(f"checkpoint not found: {checkpoint_path}; run with -train first")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded checkpoint from {checkpoint_path.resolve()}")

    model.eval()
    test(model, cfg)


if __name__ == "__main__":
    main()
