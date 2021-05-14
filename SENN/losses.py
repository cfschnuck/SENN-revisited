# torch
import torch
import torch.nn.functional as F


def simple_robustness_loss(x, aggregates, concepts, relevances):
    """Robustness Loss

    Args:
        x: image (b, n_channels, h, w)
        aggregates: aggregated relevances and concepts
        concepts: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)

    Returns:
        lipschitz like robustness loss
    """
    J_con = compute_jacobian(x, concepts)
    J_agg = compute_jacobian(x, aggregates)
    loss = J_agg - torch.bmm(J_con, relevances)
    return loss.norm(p="fro")

def mse_concepts_sparsity_loss(x, x_reconstructed, concepts, sparsity_reg, norm="l1"):
    """MSE sparsity loss

    Mean squared error loss which enforces sparsity of latent vector (concepts) by adding l1-, l2- or l-inf-norm of latent vector.

    Args:
        x: image (b, n_channels, h, w)
        x_reconstructed: reconstructed image (b, n_channels, h, w)
        concepts: concept vector (b, n_concepts)
        sparsity_reg: sparsity regularization parameter
        norm: l1-, l2-, or l-infinity-norm. Defaults to "l1".

    Returns:
        MSE sparsity loss
    """
    norms = {
        "l1" : 1, 
        "l2" : 2,
        "inf" : float("inf")
    }
    return F.mse_loss(x_reconstructed, x.detach()) + sparsity_reg * torch.norm(concepts, p=norms[norm])

def kl_div(mean, log_var):
    """Kl divergence

    Args:
        mean: mean vector
        log_var: log variance vector

    Returns:
        kl divergence
    """
    loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).mean(dim=0)
    return loss.sum()

def disentangle_loss(e1, pred1, e2, pred2):
    return F.mse_loss(e1, pred1) + F.mse_loss(e2, pred2) 

def compute_jacobian(x, fx):
    """Function to compute jacobian

    Args:
        x: 
        fx: 

    Returns:
        Jacobian
    """
    b = x.size(0) 
    m = fx.size(-1)
    J = []
    for i in range(m):
        grad = torch.zeros(b, m)
        grad[:,i] = 1.
        grad = grad.to(x.device)
        g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0]
        J.append(g.view(x.size(0),-1).unsqueeze(-1))
    J = torch.cat(J,2)
    return J