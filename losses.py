"""
Robust training losses. Based on code from
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def trades_loss(model, x_natural, y,  optimizer, args, class_weights, batch_indices=None, memory_dict=None):   
    """The TRADES KL-robustness regularization term proposed by
    Zhang et al., with added support for stability training and entropy
    regularization, and numerical stability improvements."""
    if args is not None:
        step_size, perturb_steps, epsilon = args.pgd_step_size, args.pgd_num_steps, args.epsilon
        beta = args.beta
    loss_dict = {}
    # Define KL-loss with 'sum' reduction
    criterion_kl = nn.KLDivLoss(reduction='sum')
    # Switch to eval mode to freeze batchnorm stats
    model.eval()
    batch_size = len(x_natural)
    ## Data masking
    class_weights_mask = torch.zeros(len(y)).cuda()
    for i in range(args.n_class):
        cur_indices = np.where(y.detach().cpu().numpy() == i)[0]
        class_weights_mask[cur_indices] = class_weights[i]
    # Generate adversarial example
    x_adv = x_natural.detach() + 0.  # copy tensor
    x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()  # add small noise
    logits_nat = model(x_natural)

    # Adversarial perturbation
    for i in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv)
            # Added epsilon to avoid log(0) issues for numerical stability
            loss_kl = criterion_kl(F.log_softmax(logits + 1e-10, dim=1), F.softmax(logits_nat, dim=1) + 1e-10)

        # Compute gradients
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]

        # Update x_adv with class-weighted step size and sign of gradients
        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1)* torch.sign(grad.detach())   

        # Ensure the adversarial perturbations are within epsilon bounds
        x_adv = torch.min(
            torch.max(x_adv, x_natural - (class_weights_mask * epsilon).view(-1, 1, 1, 1)),
            x_natural + (class_weights_mask * epsilon).view(-1, 1, 1, 1)
        )

        # Clamp x_adv to be within valid data range (e.g., [0, 1] for image data)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    # Switch back to training mode
    model.train()
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss()
    logits_adv = model(x_adv)
    logits_nat = model(x_natural)
    
    # Natural loss (CrossEntropy) with class weights applied
    loss_natural = (torch.nn.CrossEntropyLoss(reduction='none')(logits_nat, y)* class_weights_mask).mean()
    loss_dict['natural'] = loss_natural.item()

    # Robustness loss (KL-Divergence) with added epsilon for stability
    p_natural = F.softmax(logits_nat, dim=1) + 1e-10  # add epsilon to avoid log(0)
    loss_robust = criterion_kl(F.log_softmax(logits_adv + 1e-10, dim=1), p_natural) / (batch_size + 1e-10)  # divide with epsilon for stability

    loss_dict['robust'] = loss_robust.item()

    # Total loss: natural loss + beta * robust loss
    loss = loss_natural + beta * loss_robust

    # Save the statistics for calculating class weights after warmup
    if memory_dict is not None:
        memory_dict['probs'][batch_indices] = F.softmax(logits_adv, dim=1).detach().cpu().numpy()
        memory_dict['labels'][batch_indices] = y.detach().cpu().numpy()

    return loss, loss_dict

def mixed_trades_loss(model, x_natural, y, optimizer, args, class_weights, 
                          y_t=None, batch_indices=None, memory_dict=None):
    """
    Mixed targeted + untargeted TRADES loss with EMA smoothing during PGD.
    Applies targeted attacks to overperforming classes and untargeted to underperforming,
    using per-sample KL and cross-entropy loss, with EMA smoothing for PGD stability.
    """

    step_size, perturb_steps, epsilon = args.pgd_step_size, args.pgd_num_steps, args.epsilon
    beta = args.beta
    n_class = args.n_class
    batch_size = len(x_natural)
    device = x_natural.device

    loss_dict = {}

    # --- Class weight masking
    class_weights_mask = torch.zeros_like(y, dtype=torch.float, device=device)
    for i in range(n_class):
        class_weights_mask[y == i] = class_weights[i]

    # --- Initial adversarial perturbation
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    model.eval()
    logits_nat = model(x_natural).detach()

    tar_weights = calculate_class_weights_from_clean(
        probs=F.softmax(logits_nat, dim=1).detach().cpu().numpy(),
        labels=y.detach().cpu().numpy(),
        num_classes=args.n_class
    )
    tar_weights_mask = torch.zeros_like(y, dtype=torch.float, device=device)
    for i in range(n_class):
        idx = (y == i)
        tar_weights_mask[idx] = tar_weights[i]

    use_targeted_mask = (tar_weights_mask < tar_weights_mask.mean()).float()

    # --- Print number of targeted and untargeted samples
    num_targeted = int(use_targeted_mask.sum().item())
    num_untargeted = int((1 - use_targeted_mask).sum().item())
    #print(f"[Batch Info] Targeted samples: {num_targeted}, Untargeted samples: {num_untargeted}")

    # --- Generate y_t ≠ y for targeted samples
    rand_y_t = torch.randint(low=0, high=n_class, size=y.shape, device=device)
    mask_same = rand_y_t == y
    while mask_same.any():
        rand_y_t[mask_same] = torch.randint(low=0, high=n_class, size=(mask_same.sum().item(),), device=device)
        mask_same = rand_y_t == y
    y_t = y.clone()
    y_t[use_targeted_mask.bool()] = rand_y_t[use_targeted_mask.bool()]

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits_adv = model(x_adv)

            # Compute per-sample CE for targeted
            loss_targeted = F.cross_entropy(logits_adv, y_t, reduction='none')  # (B,)

            # Compute per-sample KL for untargeted
            p_nat = F.softmax(logits_nat, dim=1).clamp(min=1e-10)  # (B, C)
            logp_adv = F.log_softmax(logits_adv + 1e-10, dim=1)
            loss_untargeted = (p_nat * (p_nat.log() - logp_adv)).sum(dim=1)  # (B,)

            # Combine per-sample losses
            mixed_loss = use_targeted_mask * loss_targeted + (1 - use_targeted_mask) * loss_untargeted
            loss = mixed_loss.mean()  # scalar

        # Gradient update
        grad = torch.autograd.grad(loss, [x_adv])[0]

        direction = -torch.sign(grad) * use_targeted_mask.view(-1, 1, 1, 1) + \
                     torch.sign(grad) * (1 - use_targeted_mask).view(-1, 1, 1, 1)

        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1) * direction
        x_adv = torch.max(torch.min(
            x_adv,
            x_natural + (class_weights_mask.view(-1, 1, 1, 1) * epsilon)),
            x_natural - (class_weights_mask.view(-1, 1, 1, 1) * epsilon)
        )
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    x_adv = x_adv.detach()

    # --- Final loss computation
    logits_adv = model(x_adv)
    logits_nat = model(x_natural)

    # Natural loss (weighted CE)
    loss_natural = (F.cross_entropy(logits_nat, y, reduction='none')* class_weights_mask).mean()
    loss_dict['natural'] = loss_natural.item()

    # Robust loss (KL divergence between adversarial and clean predictions)
    p_nat_final = F.softmax(logits_nat, dim=1).clamp(min=1e-10)
    logp_adv_final = F.log_softmax(logits_adv + 1e-10, dim=1)
    loss_robust = (p_nat_final * (p_nat_final.log() - logp_adv_final)).sum(dim=1).mean()
    loss_dict['robust'] = loss_robust.item()

    # Also log the count of targeted/untargeted in the dict
    loss_dict['num_targeted'] = num_targeted
    loss_dict['num_untargeted'] = num_untargeted

    total_loss = loss_natural + beta * loss_robust

    # --- Memory logging
    if memory_dict is not None and batch_indices is not None:
        memory_dict['probs'][batch_indices] = F.softmax(logits_adv, dim=1).detach().cpu().numpy()
        memory_dict['labels'][batch_indices] = y.detach().cpu().numpy()
    return total_loss, loss_dict  

def madry_loss(model, x_natural, y, optimizer, args, class_weights, batch_indices=None, memory_dict=None):

    if args is not None:
        step_size, perturb_steps, epsilon = args.pgd_step_size, args.pgd_num_steps, args.epsilon
    
    criterion_ce = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_dict = {}

    ## dafa masking
    class_weights_mask = torch.zeros(len(y)).cuda()
    for i in range(args.n_class):
        cur_indices = np.where(y.detach().cpu().numpy() == i)[0]
        class_weights_mask[cur_indices] = class_weights[i]
    
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_ce = criterion_ce(logits, y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1) * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - (class_weights_mask * epsilon).view(-1, 1, 1, 1)) , 
                                           x_natural + (class_weights_mask * epsilon).view(-1, 1, 1, 1))
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    model.train()
    optimizer.zero_grad()

    logits = model(x_adv)
    
    loss_robust = (torch.nn.CrossEntropyLoss(reduction='none')(logits, y) * class_weights_mask).mean()
    loss = loss_robust
    
    loss_dict['robust'] = loss_robust.item()

    loss_dict['total'] = loss.item()

    if memory_dict is not None:
        memory_dict['probs'][batch_indices] = F.softmax(logits, dim=1).detach().cpu().numpy()
        memory_dict['labels'][batch_indices] = y.detach().cpu().numpy()

    return loss, loss_dict

import numpy as np

def calculate_class_weights_from_clean(probs, labels,num_classes, lamb=1.0, epsilon=1e-8):
    """
    Calculate adaptive class weights based on class-wise similarity of predicted probabilities.

    Args:
        probs (np.ndarray): Array of shape (N, C) with predicted class probabilities for N samples and C classes.
        labels (np.ndarray): Array of shape (N,) with ground-truth class labels.
        lamb (float): Regularization strength for weighting updates.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        np.ndarray: Array of shape (C,) with computed class weights.
    """
    class_similarity = np.zeros((num_classes, num_classes))
    class_weights = np.ones(num_classes)

    #print(num_classes)
    # Compute average class probability vectors
    for i in range(num_classes):
        cur_indices = np.where(labels == i)[0]
        if len(cur_indices) > 0:
            class_similarity[i] = np.mean(probs[cur_indices], axis=0)

    # Ensure that diagonal values are not zero to avoid instability
    class_similarity += epsilon * np.eye(num_classes)

    # Compute class weights based on inter-class similarity
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            if class_similarity[i, i] < class_similarity[j, j]:
                class_weights[i] += lamb * class_similarity[i, j] * class_similarity[j, j]
            else:
                class_weights[i] -= lamb * class_similarity[j, i] * class_similarity[i, i]

    return class_weights

