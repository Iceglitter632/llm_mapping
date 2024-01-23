import torch
import torch.nn.functional as F

class InputEncoder:
    def __init__(self, args):
        self.device = args.device
        
    def get_ground_truth():
        pass
    
    def get_onehot():
        pass
    
    def translate(batch_feature_vectors, embeddings, temperature=1.0):
        batch_size, seq_len, embedding_dim = batch_feature_vectors.shape
        # Normalize the embedding matrix
        embedding_matrix_norm = F.normalize(embeddings, dim=1)

        batch_feature_vector_norm = F.normalize(batch_feature_vectors, dim=2)
        cosine_similarities = torch.matmul(batch_feature_vector_norm, embedding_matrix_norm.T)
        sfmx = torch.softmax(cosine_similarities/temperature, dim=2)
        closest_tokens = torch.argmax(sfmx, dim=2)
        
        embeds = torch.matmul(sfmx, embeddings)

        return embeds, cosine_similarities, closest_tokens
    
    
    '''
    use translate2 if GPU is out of memory, else use translate
    '''
    def translate2(self, batch_feature_vectors, embeddings):
        batch_size, seq_len, embedding_dim = batch_feature_vectors.shape
        
        closest_tokens = torch.zeros((batch_size, seq_len), dtype=torch.float).to(device)
        embeds = torch.zeros((batch_size, seq_len, embeddings.size(1)), dtype=torch.float, requires_grad=True).to(device)
        
        embedding_matrix_norm = F.normalize(embeddings, dim=1)
        
        for i in range(batch_size):
            
            feature_vectors_norm = F.normalize(batch_feature_vectors[i], dim=1)
            cosine_similarities = torch.matmul(feature_vectors_norm, embedding_matrix_norm.T)
            logits_softmax = torch.softmax(cosine_similarities/temperature, dim=1)
            # Find the token with the highest similarity for each feature vector
            
            closest_tokens[i] = torch.argmax(logits_softmax, dim=1)
            embeds[i] = torch.matmul(logits_softmax, embeddings)

    return embeds, cosine_similarities, closest_tokens