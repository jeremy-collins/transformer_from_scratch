import torch
import torch.nn.functional as f

def scaled_dot_product_attention(query, key, value):
    # matrix multiplying on non batch dims
    attention = query.bmm(key.transpose(1, 2))
    
    # d_k = sqrt of num_features
    scale = query.size(-1) ** 0.5
    
    # normalize with softmax
    attention_scaled = f.softmax(attention / scale, dim=1)
    
    return attention_scaled.bmm(value)
    