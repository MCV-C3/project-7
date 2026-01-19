import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        ce_loss = self.ce(student_logits, labels)

        # Soft label loss
        s_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        t_probs = F.softmax(teacher_logits / self.T, dim=1)

        kd_loss = self.kl(s_log_probs, t_probs) * (self.T ** 2)

        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss
