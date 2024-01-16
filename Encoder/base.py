import torch
import torch.nn.functional as F

class InputEncoder:
    def __init__(self, args):
        self.device = args.device
        
    def get_ground_truth():
        pass
    
    def get_onehot():
        pass
    
    def translate(self, batch_feature_vectors, embeddings):
        batch_size, seq_len, embedding_dim = batch_feature_vectors.shape
        closest_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)

        # Normalize the embedding matrix
        embedding_matrix_norm = F.normalize(embeddings, dim=1)

        closest_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            # Normalize the feature vectors for the i-th sample in the batch
            feature_vectors_norm = F.normalize(batch_feature_vectors[i], dim=1)

            # Compute cosine similarity for the entire sequence at once
            cosine_similarities = torch.matmul(feature_vectors_norm, embedding_matrix_norm.T)

            # Find the token with the highest similarity for each feature vector
            closest_tokens[i] = torch.argmax(cosine_similarities, dim=1)

        return closest_tokens