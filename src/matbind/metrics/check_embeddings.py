import torch
from typing import Dict, Optional


# Based on the paper https://proceedings.mlr.press/v202/marbut23a/marbut23a.pdf
# more can be done with labelled data https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00088e
class LatentSpaceMetrics:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def local_smoothness(embeddings: torch.Tensor, k: int = 5) -> Dict[str, float]:
        """Calculate local smoothness metrics"""
        distances = torch.cdist(embeddings, embeddings)
        _, indices = torch.topk(distances, k=k + 1, dim=1, largest=False)

        local_variations = []
        for i in range(len(embeddings)):
            neighbors = embeddings[indices[i][1:]]  # exclude self
            center = embeddings[i]
            local_var = torch.var(neighbors - center, dim=0).mean()
            local_variations.append(local_var)

        local_variations = torch.stack(local_variations)

        if isinstance(local_variations, torch.Tensor):
            local_variations = local_variations.detach()

        return {
            "local_smoothness_mean": local_variations.mean().item(),
            "local_smoothness_std": local_variations.std().item(),
            "local_smoothness_max": local_variations.max().item(),
        }

    @staticmethod
    def global_structure(
        embeddings: torch.Tensor,
        original_distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Calculate global structure preservation metrics"""
        latent_distances = torch.cdist(embeddings, embeddings)

        if original_distances is not None:
            orig_norm = original_distances / original_distances.max()
            latent_norm = latent_distances / latent_distances.max()

            distance_correlation = torch.corrcoef(
                torch.stack([orig_norm.flatten(), latent_norm.flatten()])
            )[0, 1].item()
        else:
            distance_correlation = None

        return {
            "distance_correlation": float(distance_correlation)
            if distance_correlation is not None
            else 0.0,
            "mean_pairwise_distance": latent_distances.mean().item(),
            "std_pairwise_distance": latent_distances.std().item(),
        }

    @staticmethod
    def manifold_quality(embeddings: torch.Tensor, k: int = 5) -> Dict[str, float]:
        """Calculate manifold quality metrics"""
        distances = torch.cdist(embeddings, embeddings)  # l2-norm of each pair
        _, indices = torch.topk(distances, k=k + 1, dim=1, largest=False)

        condition_numbers = []
        linearity_scores = []

        for i in range(len(embeddings)):
            neighbors = embeddings[indices[i]]
            centered = neighbors - neighbors.mean(dim=0)

            try:
                U, S, V = torch.svd(centered)
                condition_number = S[0] / S[-1]
                explained_var_ratio = S[0] / S.sum()

                condition_numbers.append(condition_number)
                linearity_scores.append(explained_var_ratio)
            except:
                continue

        condition_numbers = torch.stack(condition_numbers)
        linearity_scores = torch.stack(linearity_scores)

        return {
            "condition_number_mean(ER)": condition_numbers.mean().item(),
            "condition_number_std": condition_numbers.std().item(),
            "local_linearity_mean": linearity_scores.mean().item(),
            "local_linearity_std": linearity_scores.std().item(),
        }

    @staticmethod
    def isotropy(embeddings: torch.Tensor) -> Dict[str, float]:
        """Calculate isotropy metrics"""
        centered = embeddings - embeddings.mean(dim=0)
        cov = torch.mm(centered.t(), centered) / len(embeddings)
        eigenvalues = torch.linalg.eigvalsh(cov)

        return {
            "isotropy": (eigenvalues.min() / eigenvalues.max()).item(),
            "eigenvalue_ratio": (eigenvalues[-1] / eigenvalues[0]).item(),
        }

    @staticmethod
    def interpolation_smoothness(
        z1: torch.Tensor, z2: torch.Tensor, steps: int = 50
    ) -> Dict[str, float]:
        """Measure smoothness of interpolation between points"""
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            interp = (1 - alpha) * z1 + alpha * z2
            interpolations.append(interp)

        interpolations = torch.stack(interpolations)
        dists = torch.norm(interpolations[1:] - interpolations[:-1], dim=1)

        return {
            "interp_variance": torch.var(dists).item(),
            "interp_max_jump": torch.max(dists).item(),
            "interp_mean_dist": torch.mean(dists).item(),
        }

    def eee(
        self, embeddings: torch.Tensor, explained_var_threshold: float = 0.9
    ) -> Dict[str, float]:
        """
        Calculate Early Eigenvalue Enrichment (EEE)

        EEE measures how quickly the principal components explain the variance,
        indicating how efficiently the latent space is being used.

        Args:
            embeddings: Embedding tensor
            explained_var_threshold: Threshold for cumulative explained variance (default: 0.9)

        Returns:
            Dictionary containing EEE score
        """
        embeddings = embeddings.to(self.device)

        # Center the embeddings
        centered = embeddings - embeddings.mean(dim=0)

        # Calculate covariance matrix
        cov = torch.mm(centered.t(), centered) / (len(embeddings) - 1)

        # Get eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)  # Sort in descending order

        # Calculate cumulative explained variance ratio
        total_var = eigenvalues.sum()
        cum_var_ratio = torch.cumsum(eigenvalues, 0) / total_var

        # Find number of components needed to explain threshold of variance
        n_components = torch.where(cum_var_ratio >= explained_var_threshold)[0][0] + 1

        # Calculate EEE score (normalized by dimension)
        dimension = embeddings.shape[1]
        eee_score = 1 - (n_components / dimension)

        return {
            "eee_score": eee_score.item(),
            "n_components_threshold": n_components.item(),
        }

    def vrm(self, embeddings: torch.Tensor, n_bins: int = 50) -> Dict[str, float]:
        """
        Calculate Vasicek Ratio MSE (VRM)

        VRM measures the entropy-based complexity of the latent space distribution
        using Vasicek's spacing method for entropy estimation.

        Args:
            embeddings: Embedding tensor
            n_bins: Number of bins for entropy estimation (default: 50)

        Returns:
            Dictionary containing VRM score
        """

        embeddings = embeddings.to(self.device)

        def vasicek_entropy(x: torch.Tensor, m: int) -> float:
            n = len(x)
            x_sorted = torch.sort(x)[0]
            spacing = x_sorted[m:] - x_sorted[:-m]
            return torch.log(torch.mean(spacing) * (n + 1) / (2 * m)).item()

        # Calculate entropy for each dimension
        entropies = []
        for dim in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim]
            entropy = vasicek_entropy(dim_values, max(1, len(dim_values) // n_bins))
            entropies.append(entropy)

        entropies = torch.tensor(entropies)

        # Calculate mean and variance of entropies
        mean_entropy = entropies.mean()
        var_entropy = entropies.var()

        # Calculate VRM score (ratio of variance to mean)
        vrm_score = var_entropy / (
            mean_entropy + 1e-10
        )  # Add small constant to prevent division by zero

        return {
            "vrm_score": vrm_score.item(),
            "mean_entropy": mean_entropy.item(),
            "entropy_variance": var_entropy.item(),
        }


if __name__ == "__main__":
    # Example usage
    metrics = LatentSpaceMetrics()
    # Create a mixture of delta functions at different points
    points = [0, 2, 5, 10]  # Non-uniform spacing
    probs = [0.4, 0.3, 0.2, 0.1]  # Non-uniform probabilities

    choices = torch.tensor(points)[
        torch.multinomial(torch.tensor(probs), num_samples=100 * 32, replacement=True)
    ].reshape(100, 32)
    embeddings_randn = choices.float()

    # Calculate EEE
    eee_metrics = metrics.eee(embeddings_randn, explained_var_threshold=0.95)
    print("EEE metrics:", eee_metrics)

    # Calculate VRM
    vrm_metrics = metrics.vrm(embeddings_randn, n_bins=4)
    print("VRM metrics:", vrm_metrics)

    metrics = LatentSpaceMetrics()
    # Generate new gaussian embeddings
    embeddings_normal = torch.normal(0, 1, size=(100, 32))

    print("Normalized embeddings metrics:")
    eee_metrics = metrics.eee(embeddings_normal, explained_var_threshold=0.95)
    print("EEE metrics:", eee_metrics)

    vrm_metrics = metrics.vrm(embeddings_normal, n_bins=40)
    print("VRM metrics:", vrm_metrics)
