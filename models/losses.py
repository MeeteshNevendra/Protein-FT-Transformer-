import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassFocalLoss(nn.Module):
    """Focal Loss for multiclass classification with label smoothing"""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.reduction = reduction
        
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        
        Returns:
            loss: scalar
        """
        num_classes = inputs.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / (num_classes - 1)
            
            one_hot = torch.zeros_like(inputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            one_hot = one_hot * smooth_pos + (1 - one_hot) * smooth_neg
        else:
            one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        # Calculate probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Calculate focal weights
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = -(one_hot * log_probs).sum(dim=1)
        
        # Apply focal weighting
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        # Create smoothed labels
        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / (num_classes - 1)
        
        one_hot = torch.zeros_like(inputs)
        one_hot.fill_(smoothing_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Calculate loss
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(one_hot * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
