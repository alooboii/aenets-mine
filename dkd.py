import torch
import torch.nn as nn
import torch.nn.functional as F


def dkd_loss(student_logits, teacher_logits, targets, alpha, beta, temperature):
    """
    Decoupled Knowledge Distillation loss (Zhao et al., 2022).
    https://arxiv.org/abs/2203.08679

    Decomposes the classical KD loss into two independent components:
      - TCKD: Target Class Knowledge Distillation — transfers knowledge on the
              target class probability, analogous to label smoothing.
      - NCKD: Non-Target Class Knowledge Distillation — transfers knowledge
              among the non-target classes, which is the core "dark knowledge".

    The classical KD loss is a *coupled* weighted sum of TCKD and NCKD where
    a single weight (scaled by p_t, the teacher's target-class confidence)
    ties the two together. DKD decouples them so each can be weighted
    independently via alpha and beta.

    Args:
        student_logits (Tensor): Raw student logits, shape [B, C].
        teacher_logits (Tensor): Raw teacher logits, shape [B, C].
        targets (Tensor):        Ground-truth class indices, shape [B].
        alpha (float):           Weight for TCKD loss.
        beta (float):            Weight for NCKD loss.
        temperature (float):     Softmax temperature (applied to both TCKD and NCKD).

    Returns:
        Tensor: Scalar DKD loss.
    """
    # ------------------------------------------------------------------ #
    # Build one-hot mask for the target class                             #
    # ------------------------------------------------------------------ #
    B, C = student_logits.shape
    target_mask = torch.zeros_like(student_logits).scatter_(1, targets.unsqueeze(1), 1.0)  # [B, C]
    non_target_mask = 1.0 - target_mask                                                    # [B, C]

    # ------------------------------------------------------------------ #
    # Soft probabilities at temperature T                                 #
    # ------------------------------------------------------------------ #
    s_soft = F.softmax(student_logits / temperature, dim=1)   # [B, C]
    t_soft = F.softmax(teacher_logits / temperature, dim=1)   # [B, C]

    # ------------------------------------------------------------------ #
    # TCKD — binary KL over (target class vs. all others)                #
    #                                                                     #
    # For each sample we treat the distribution as a Bernoulli:          #
    #   p_t  = prob of target class                                       #
    #   1-p_t = prob of "not target"                                      #
    # We compute KL(student_binary || teacher_binary) via cross-entropy.  #
    # ------------------------------------------------------------------ #
    # Target-class probability for student and teacher
    s_target_prob = (s_soft * target_mask).sum(dim=1, keepdim=True)      # [B, 1]
    t_target_prob = (t_soft * target_mask).sum(dim=1, keepdim=True)      # [B, 1]

    # Binary distribution: [p_t, 1-p_t]
    s_binary = torch.cat([s_target_prob, 1.0 - s_target_prob], dim=1)   # [B, 2]
    t_binary = torch.cat([t_target_prob, 1.0 - t_target_prob], dim=1)   # [B, 2]

    tckd_loss = F.kl_div(
        torch.log(s_binary + 1e-8),
        t_binary,
        reduction='batchmean'
    ) * (temperature ** 2)

    # ------------------------------------------------------------------ #
    # NCKD — KL over non-target classes only                             #
    #                                                                     #
    # Mask out the target class, then re-normalise the remaining         #
    # probabilities to form a valid distribution over C-1 classes.       #
    # ------------------------------------------------------------------ #
    # Mask target class with -inf before softmax so it contributes 0 prob
    s_non_target_logits = student_logits - 1e9 * target_mask             # [B, C]
    t_non_target_logits = teacher_logits - 1e9 * target_mask             # [B, C]

    s_non_target = F.softmax(s_non_target_logits / temperature, dim=1)  # [B, C]
    t_non_target = F.softmax(t_non_target_logits / temperature, dim=1)  # [B, C]

    nckd_loss = F.kl_div(
        torch.log(s_non_target + 1e-8),
        t_non_target,
        reduction='batchmean'
    ) * (temperature ** 2)

    return alpha * tckd_loss + beta * nckd_loss


class DKD(nn.Module):
    """
    Decoupled Knowledge Distillation (Zhao et al., 2022).
    https://arxiv.org/abs/2203.08679

    Replaces the classical KD logit loss with two independently-weighted
    components:
      - TCKD (alpha): knowledge about the target class.
      - NCKD (beta):  knowledge about non-target classes ("dark knowledge").

    The paper shows that classical KD suppresses NCKD when the teacher is
    highly confident (high p_t), while DKD allows each component to be
    weighted freely, improving student performance — especially when
    beta > alpha.

    Interface is intentionally kept compatible with the other KD wrappers in
    this repo: forward() returns (teacher_logits, student_logits, dkd_loss).
    The training loop is responsible for adding the CE classification loss.

    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network to be trained.
        alpha (float):       Weight for TCKD component (default 1.0).
        beta (float):        Weight for NCKD component (default 1.0).
        temperature (float): Softmax temperature (default 4.0).
    """

    def __init__(self, teacher, student, alpha=1.0, beta=1.0, temperature=4.0):
        super(DKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        # Freeze teacher — it is only used as a static oracle.
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, targets):
        """
        Args:
            x (Tensor):       Input batch, shape [B, C, H, W].
            targets (Tensor): Ground-truth labels, shape [B].  Required by
                              DKD to build the target-class mask.

        Returns:
            tuple: (teacher_logits, student_logits, dkd_loss)
                   dkd_loss is a scalar Tensor — add your CE loss on top in
                   the training loop.
        """
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        loss = dkd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )

        return teacher_logits, student_logits, loss