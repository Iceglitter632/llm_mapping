import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2:
    def __init__(self, args):
        
        # Load pre-trained GPT-2 model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(args.llm)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.llm)

        self.embeddings = self.model.lm_head.weight
        # embedding_matrix = model.transformer.wte.weight
        self.feature_dim = self.model.config.hidden_size
        self.vocab_len = self.model.config.vocab_size
        
        self.device = args.device
        self.model.to(self.device)
    
    def forward_with_embeddings(self, embeddings):
        """
        Forward pass through GPT-2 for sequential token prediction logits from embeddings.

        :param embeddings: Embeddings of the sequence, shape [batch_size, seq_len, embedding_dim].
        :return: Tensor of logits for token predictions, shape [batch_size, seq_len, vocab_size].
        """
        
        batch_size, seq_len, _ = embeddings.size()
        predicted_logits = torch.zeros((batch_size, seq_len, self.vocab_len), device=self.device)
        
        self.model.eval()

        embeddings = embeddings.detach()
        
        for i in range(seq_len):
            # Use embeddings up to the i-th position to predict the next token
            input_embeddings = embeddings[:, :i+1, :]

            # Forward pass through GPT-2
            with torch.no_grad():
                outputs = self.model(inputs_embeds=input_embeddings)
            logits = outputs.logits

            # Get the logits for the next position (i+1)
            predicted_logits[:, i, :] = logits[:, -1, :]  # Last token in the sequence

        predicted_logits.requires_grad = True

        return predicted_logits
    
    def find_closest_token_cosine(self, batch_feature_vectors):
        """
        Find the LLM token whose embedding is closest to the given feature vector.

        :param feature_vector: The feature vector (from the mapper). Shape: (embedding_dim,)
        :param embedding_matrix: LLMs embedding matrix. Shape: (vocab_size, embedding_dim)
        :return: The ID of the closest token.
        """
        batch_size, seq_len, embedding_dim = batch_feature_vectors.shape
        closest_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        # Normalize the embedding matrix
        embedding_matrix_norm = F.normalize(self.embeddings, dim=1)

        closest_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i in range(batch_size):
            # Normalize the feature vectors for the i-th sample in the batch
            feature_vectors_norm = F.normalize(batch_feature_vectors[i], dim=1)

            # Compute cosine similarity for the entire sequence at once
            cosine_similarities = torch.matmul(feature_vectors_norm, embedding_matrix_norm.T)

            # Find the token with the highest similarity for each feature vector
            closest_tokens[i] =torch.argmax(cosine_similarities, dim=1)

        return closest_tokens
    
    def find_closest_token_logits(self, batch_feature_vectors):
        """
        Find the LLM token who has the highest dot product to the given feature vector
        This acts as the action of our REINFORCE algorithm
        
        return prob: Probability of each token getting chosen => shape:(batch_size, seq_len, llm_vocabulary_size)
        return closest_tokens: The token with the highest probability => shape:(batch_size, seq_len) 
        """
        dot_product = torch.matmul(batch_feature_vectors, self.embeddings.T)
        probs = F.softmax(dot_product, dim=-1)
        closest_tokens = torch.argmax(probs, dim=-1)
        
        return dot_product, closest_tokens

    def generate_next_token_predictions(self, token_sequences):
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids=token_sequences, output_hidden_states=True)
        
        return outputs.hidden_states[-1]
            
        
    
    def get_ground_truth(self, mapped_feature_vector):
    
        action_logits, ground_truth = self.find_closest_token_logits(mapped_feature_vector)

        return action_logits, ground_truth