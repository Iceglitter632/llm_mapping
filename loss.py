import torch
import torch.nn.functional as F

def Reinforce_Loss(logits, targets, loss, gamma=1.0, device="cpu"):
    """
    Calculate the REINFORCE loss for sequence prediction.

    :param logits: Logits from the model, shape [batch_size, seq_len, vocab_size].
    :param targets: Ground truth sequence, shape [batch_size, seq_len].
    :param rewards: Reward for each step in the sequence, shape [batch_size, seq_len].
    :param gamma: Discount factor for future rewards.
    :return: The REINFORCE loss (to be maximized).
    """
    batch_size, seq_len, _ = logits.shape

    # return loss / seq_len
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_targets = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    # Create a discount matrix
    discount_matrix = torch.zeros((seq_len, seq_len)).to(device)

    # Fill the matrix according to the given pattern
    for i in range(seq_len):
        for j in range(i, seq_len):
            discount_matrix[i, j] = gamma ** (j - i)


    # Calculate discounted rewards
    discounted_loss = loss.unsqueeze(1) * discount_matrix
    cumulative_loss = discounted_loss.sum(dim=2)
    
    # Calculate loss
    # total_loss = -torch.sum(log_probs_targets * cumulative_loss) / batch_size / seq_len
    total_loss = torch.sum(log_probs_targets * cumulative_loss) / batch_size / seq_len

    return total_loss

def CrossEntropySG_Loss(llm, mapped_feature_vector, targets, reduction='mean'):
    """
    Custom cross-entropy loss with straight-through estimator.
    :return: Loss value.
    """
    batch_size, seq_len, embedding_dim = mapped_feature_vector.shape
    
    # Closest tokens have shape [batch_size, seq_len]
    # closest_tokens = get_llm_ground_truth(mapped_feature_vector)

    closest_embeddings = llm.embeddings[targets]
    closest_embeddings = closest_embeddings.reshape(batch_size, seq_len, embedding_dim)

    # STE_LOGITS have shape [batch_size, seq_len, embedding_dim]
    ste_logits = (closest_embeddings - mapped_feature_vector.detach()) + mapped_feature_vector

    predictions = llm.forward_with_embeddings(ste_logits)
    predictions = predictions.reshape(batch_size*seq_len, -1)
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(predictions, targets, reduction=reduction)

    return loss
