import torch

def test_retrieval_function():
    embeddings_central_modality = torch.randn(512, 128)  # Example embeddings for central modality
    embeddings_other_modality = torch.randn(512, 128)  # Example embeddings for other modality
    batch_size = embeddings_central_modality.shape[0]
    chunk_size = min(512, batch_size)
    cos_sim_list=[]
    for i in range(0, batch_size, chunk_size):
        chunk_end = min(i + chunk_size, batch_size)
        chunk_central = embeddings_central_modality[i:chunk_end]
        chunk_other = embeddings_other_modality
        chunk_cos_sim = (chunk_central.unsqueeze(1) * chunk_other.unsqueeze(0)).sum(dim=2)
        cos_sim_list.append(chunk_cos_sim.cpu())  # Move to CPU immediately

    cos_sim = torch.cat(cos_sim_list, dim=0)

    flatten_cos_sim = cos_sim.flatten()
    
    cos_sim_new = torch.matmul(
            embeddings_central_modality, embeddings_other_modality.T
        )  # Shape: [batch_size, batch_size]

    print((cos_sim_new - cos_sim)[cos_sim_new - cos_sim > 1e-5])
    assert torch.allclose(cos_sim, cos_sim_new, atol = 1e-5), "Cosine similarity matrices do not match"
    

