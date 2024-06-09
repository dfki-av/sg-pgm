import torch
import torch.nn as nn

def sinkhorn_rpm(log_alpha, n_iters: int = 5, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
    log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

    for i in range(n_iters):
        # Row normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)
        if eps > 0:
            if prev_alpha is not None:
                abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                    break
            prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

    return torch.exp(log_alpha_padded)#[:, :-1, :-1]