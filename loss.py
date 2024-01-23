import torch
import torch.nn.functional as F

def Reinforce_Loss(logits, translated, loss, discount_matrix, normalize_factor, gamma=1.0, alpha=1, temperature=temperature, device="cpu"):
    """
    Calculate the REINFORCE loss for sequence prediction.

    :param logits: Logits from the model, shape [batch_size, seq_len, vocab_size].
    :param targets: Ground truth sequence, shape [batch_size, seq_len].
    :param rewards: Reward for each step in the sequence, shape [batch_size, seq_len].
    :param gamma: Discount factor for future rewards.
    :return: The REINFORCE loss (to be maximized).
    """
    batch_size, seq_len, _ = logits.shape
    translated = translated.to(torch.int64)
    
    # return loss / seq_len
    log_probs = F.log_softmax(logits/temperature, dim=-1)
    log_probs_targets = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    discounted_loss = loss.unsqueeze(1) * discount_matrix
    cumulative_loss = discounted_loss.sum(dim=2) / normalize_factor / alpha
    
    # Calculate loss
    total_loss = torch.sum(log_probs_targets * cumulative_loss) / batch_size / seq_len 
    
    return total_loss
