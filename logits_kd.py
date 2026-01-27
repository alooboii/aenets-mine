import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitKD(nn.Module):
    """
    Traditional Knowledge Distillation using only output logits.
    
    This is the original knowledge distillation method proposed by Hinton et al.
    that only uses the soft targets from teacher's output logits without
    intermediate feature matching.
    
    Args:
        teacher (nn.Module): Pretrained teacher network with attribute `.model`.
        student (nn.Module): Student network with attribute `.model`.
    """
    def __init__(self, teacher, student):
        super(LogitKD, self).__init__()
        self.teacher = teacher
        self.student = student
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass that computes both teacher and student outputs.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            tuple: (teacher_logits, student_logits, None)
                   The third element is None to maintain compatibility with other methods.
        """
        # Forward pass through teacher and student networks
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        
        # Return None for the third element to maintain compatibility
        # with the training loop that expects 3 or 4 return values
        return teacher_logits, student_logits, None