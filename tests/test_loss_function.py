from httpx import post
import torch

def test_info_NCE():
    from matbind.model.losses.info_nce import InfoNCELoss, NCE
    from matbind.model.components.postprocessing import Postprocessor, LearnableLogitScaling
    
    # Test with dummy inputs
    z_i = torch.randn(2, 2)  # Two samples with two features each
    z_j = torch.randn(2, 2)  # Two samples with two features each
    temperature = 0.07

    package_loss = InfoNCELoss(
        temperature=temperature,
        symmetric=True,
        sparse=False,
        l1_norm_weight=0.01,
        reduction="mean",
        negative_mode="unpaired"
    )

    package_loss = package_loss(z_i, z_j)
    
    self_loss = NCE(
        symmetric=True,
        sparse=False,
        l1_norm_weight=0.01,
        reduction="mean"
    )

    postprocessor = Postprocessor(
        log_scalor=LearnableLogitScaling(learnable=False, logit_scale_init=temperature),
        normalize=True
    )
    normalizer = Postprocessor(
        log_scalor=None,
        normalize=True
    )
    z_i_processed = normalizer(z_i)
    z_j_processed = postprocessor(z_j)
    self_loss = self_loss(z_i_processed, z_j_processed)

    # Check if the loss is non-negative
    assert package_loss == self_loss, "Losses do not match"

def test_cosine_similarity():
    
    # Test with dummy inputs
    z_i = torch.randn(512, 2)  # Two samples with two features each
    z_j = torch.randn(1024, 2)  # Two samples with two features each

    cosine_similair = torch.nn.functional.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
    print(cosine_similair.shape)
    direct_calculate = (torch.nn.functional.normalize(z_i, dim=-1).unsqueeze(1) * torch.nn.functional.normalize(z_j, dim=-1).unsqueeze(0)).sum(dim=2)
    assert cosine_similair.shape == direct_calculate.shape, "Cosine similarity shape mismatch"
    assert torch.all(cosine_similair == direct_calculate)
