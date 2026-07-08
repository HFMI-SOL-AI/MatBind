from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from mp_api.client import MPRester
from sklearn.manifold import TSNE
import numpy as np
import torch
import umap
import pickle as pkl
from torch import nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class ModalityEmbeddings:
    """Stores embeddings for a modality with their corresponding material IDs and indices."""

    def __init__(self, embeddings: torch.Tensor, material_ids: list[str], indices: list[int]):
        self.embeddings = embeddings
        self.material_ids = material_ids
        self.indices = indices

    def __len__(self):
        return len(self.material_ids)

crystal_system_mapping = {
    "Cubic": 0,
    "Tetragonal": 1,
    "Orthorhombic": 2,
    "Hexagonal": 3,
    "Trigonal": 4,
    "Monoclinic": 5,
    "Triclinic": 6,
}

# file path
PATH = "/home/l.yang/sol-ai/matbind/solai_embeddings"
# load embedding
# embeddings = pd.read_pickle(f"{PATH}/text_crystal_pxrd_dos_20250130_1739_embeddings_cpu.pkl")
embeddings = None
# load validation dataset
# validation_dataset = pd.read_pickle(f"{PATH}/valid_data.pkl")
validation_dataset = None

def check_entry():
    # print(embeddings['dos'])
    print(validation_dataset.iloc[2030]["crystal_structure"])
    # print(validation_dataset.shape)

def get_other_property(validation_data):
    material_ids = validation_data["material_id"]
    print(len(material_ids))
    with MPRester("GaFvrGA8ychgbZvSh67ntMP9K0Q16Pg4") as mpr:
        docs = mpr.materials.summary.search(
            material_ids=material_ids.to_list(),fields=["material_id","formation_energy_per_atom"]
        )
    # print(docs[0].symmetry.crystal_system)
    material_ids = [doc.material_id for doc in docs]
    formation_energy = [doc.formation_energy_per_atom for doc in docs]
    new_df = pd.DataFrame({"material_id":material_ids, "formation_energy":formation_energy})
    validation_data = pd.merge(validation_data,new_df,on="material_id", how="outer")
    # print(validation_data["formation_energy"])
    return validation_data

def svd(X, n_components:int = 2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]

def tsne():
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['DOS', 'Crystal Structure', 'Text', 'PXRD']

    dos_embedding = F.normalize(embeddings['dos'].float()).numpy()
    crystal_embedding = F.normalize(embeddings['crystal_structure'].float()).numpy()
    text_embedding = F.normalize(embeddings['text'].float()).numpy()
    pxrd_embedding = F.normalize(embeddings['pxrd'].float()).numpy()
    label = np.concatenate((np.ones(dos_embedding.shape[0]) * 0,
                            np.ones(crystal_embedding.shape[0]),
                            np.ones(text_embedding.shape[0])*2,
                            np.ones(pxrd_embedding.shape[0])*3))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all = np.concatenate((dos_embedding, crystal_embedding, text_embedding, pxrd_embedding))
    X = tsne.fit_transform(all)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for i, (color, label_name) in enumerate(zip(colors, labels)):
        mask = label == i
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=label_name, alpha=0.7)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.set_title('t-SNE Visualization of Different Embedding Types')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, alpha=0.3)
    plt.savefig("tsne.png")

def dos_umap():
    validation = get_other_property(validation_dataset)
    dos_embedding = embeddings['crystal_structure'].float().numpy()
    crystal_embedding = embeddings['crystal_structure'].float().numpy()
    pxrd_embedding = embeddings['pxrd'].float().numpy()
    label = validation['formation_energy']
    reducer = umap.UMAP(n_components=2)
    X = reducer.fit_transform(dos_embedding)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    scatter = ax.scatter(X[:, 0], X[:, 1],c=label)
    plt.colorbar(scatter)
    plt.show()

def text_tsne():
    embeddings_path = Path("/p/project1/solai/yang21/MatBind/experiments/crys_pxrd_dos_text_partial/crystal/all/embeddings.pkl")
    with embeddings_path.open("rb") as f:
        data = pkl.load(f)

    embeddings_dict = {}
    for modality, emb_tensor in data["embeddings"].items():
        metadata = data["metadata"][modality]
        embeddings_dict[modality] = ModalityEmbeddings(
            embeddings=emb_tensor,
            material_ids=metadata["material_ids"],
            indices=metadata["indices"],
        )
    text_embedding = F.normalize(embeddings_dict["text"].embeddings, dim=-1).numpy()
    material_ids = [str(mid).strip() for mid in embeddings_dict["text"].material_ids]

    # property_df = pd.read_csv("/p/project1/solai/datasets/materials_project/symmetry_data.csv")
    property_df = pd.read_parquet("/p/project1/solai/datasets/materials_project/dos.parquet")
    property_df["material_id"] = property_df["material_id"].astype(str).str.strip()

    crystal_lookup = (
        property_df.drop_duplicates(subset=["material_id"], keep="first")
        .set_index("material_id")["band_gap"]
    )
    crystal_labels = crystal_lookup.reindex(material_ids)

    valid_mask = crystal_labels.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("No crystal_system labels matched text embedding material_ids")

    text_embedding = text_embedding[valid_mask]
    band_gap_values = crystal_labels[valid_mask].astype(float).to_numpy()

    focus_min, focus_max = 0.0, 2.5
    plot_band_gap = np.clip(band_gap_values, focus_min, focus_max)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X = tsne.fit_transform(text_embedding)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=plot_band_gap, s=1, cmap="viridis", vmin=focus_min, vmax=focus_max)
    cbar = plt.colorbar(scatter, extend="max")
    cbar.set_label("Band gap (eV), clipped at 2.5")
    plt.savefig("text_band_gap.png")

def inter_arithmetic(embedding1, embedding2, operation = "plus"):
    embedding1 = nn.functional.normalize(embedding1, dim=-1)
    embedding2 = nn.functional.normalize(embedding2, dim=-1)
    if operation == "plus":
        return embedding1*0.5 + embedding2*0.5
    else:
        return embedding1 - embedding2

def cosine_nearest(embedding, dataset):
    similarity = cosine_similarity(embedding.float().numpy(), dataset.float().numpy())
    indexes = np.argsort(similarity[0])[::-1][:3]
    return indexes, similarity[0]

def inter(used_embedding:str = "dos"):
    # generate two indexes for embedding
    dataset = embeddings[used_embedding]
    another = embeddings["pxrd"]
    target_dataset = embeddings["crystal_structure"]
    valid = validation_dataset
    indexes = np.random.randint(low = 0, high = len(dataset), size = 2)
    # indexes = [886, 2030]
    print(indexes)
    candidate = inter_arithmetic(dataset[indexes[0]][None:,], another[indexes[1]][None,:])
    candidate_indexes, similarity = cosine_nearest(candidate, target_dataset)
    candidate_indexes = candidate_indexes[np.logical_and(candidate_indexes != indexes[0], candidate_indexes != indexes[1])]
    candidate_index = candidate_indexes[0]
    print(candidate_index)
    print(similarity[candidate_index])
    print(valid["material_id"].iloc[indexes])
    print(valid["material_id"].iloc[candidate_index])
# evaluation ?

def intra():
    dos_embeddings = embeddings["dos"][validation_dataset['band_gap']>0]
    crystal_embeddings = embeddings["crystal_structure"][validation_dataset['band_gap']>0]
    valid = validation_dataset[validation_dataset['band_gap']>0]

    indexes = np.random.randint(low = 0, high = len(dos_embeddings), size = 2)
    print(indexes)
    diff = inter_arithmetic(dos_embeddings[indexes[0]][None:,], dos_embeddings[indexes[1]][None,:], "minus")
    candidate = inter_arithmetic(diff, crystal_embeddings[indexes[1]][None,:])

    candidate_indexes, similarity = cosine_nearest(candidate, crystal_embeddings)
    # candidate_indexes = candidate_indexes[np.logical_and(candidate_indexes != indexes[0], candidate_indexes != indexes[1])]
    candidate_index = candidate_indexes[0]
    candidate_embedding = crystal_embeddings[candidate_index]
    candidate_embedding = nn.functional.normalize(candidate_embedding, dim=-1)
    embedding_0 = nn.functional.normalize(crystal_embeddings[indexes[1]][None,:],dim=-1)
    similarity_0 = embedding_0 @ candidate_embedding.T
    print(candidate_index)
    print(similarity[candidate_index])
    print(similarity_0)
    print(valid["material_id"].iloc[indexes])
    print(valid["material_id"].iloc[candidate_index])

def evaluation(candidate:str = "dos",k:int = 2):
    dos_embeddings = embeddings[candidate]
    crystal_embeddings = embeddings["crystal_structure"]
    ground_truth_b_indices = torch.arange(4343).unsqueeze(1).repeat(1,4343)
    # print(ground_truth_b_indices)

    diff = dos_embeddings[:, None, :] - dos_embeddings[None, :, :]
    candidate_embedding = diff + crystal_embeddings[None,: ,:]

    result_flat = candidate_embedding.view(-1, 128).cuda()  # (4343*4343, 128)
    crystal_embeddings = crystal_embeddings.view(-1, 128).cuda()
    N = result_flat.shape[0]

    print("Computing similarities...")
    result_norm = F.normalize(result_flat, dim=1)  # [N, D]
    crystal_embeddings_norm = F.normalize(crystal_embeddings, dim=1)  # [M, D]

    # Choose a batch size that fits your memory
    total_matches = 0
    total_queries = 0
    batch_size = 5000

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = result_norm[start:end]  # [B, 128]

        # Compute cosine similarity to all B vectors
        sims = batch @ crystal_embeddings_norm.T  # [B, 4343]

        # Get max similarity and indices
        scores, indices = torch.topk(sims, k=k, dim=1)  # [B, k]

        # Get ground truth for this batch
        batch_ground_truth = ground_truth_b_indices.view(-1)[start:end]  # [B]

        # Check if ground truth is in top-k for each query in batch
        matches = (indices.cpu() == batch_ground_truth.unsqueeze(1)).any(dim=1)  # [B]

        # Accumulate results
        total_matches += matches.sum().item()
        total_queries += matches.shape[0]

    top_k_accuracy = total_matches / total_queries
    return top_k_accuracy

def full_evaluation():
    results = {"dos":[], "pxrd":[], "crystal_structure":[], "text":[]}
    candiate_embeddings = ["dos", "pxrd", "crystal_structure", "text"]
    top_k = [1, 5, 10]
    for candiate, k in tqdm([(a,b) for a in candiate_embeddings for b in top_k]):
        result = evaluation(candiate, k)
        results[candiate].append(result)
    print(results)

def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0

        acc = (lvm_mask * llm_mask).sum(dim=1) / topk

        return acc.mean().item()

def evaluate_mutual_knn():
    candiate_embeddings = ["dos", "pxrd", "crystal_structure", "text"]
    results = {"dos":[], "pxrd":[], "crystal_structure":[], "text":[]}
    for feature_a, feature_b in tqdm([(a,b) for a in candiate_embeddings for b in candiate_embeddings]):
        result = mutual_knn(embeddings[feature_a], embeddings[feature_b], 5)
        results[feature_a].append(result)
    print(results)

if __name__ == "__main__":
    text_tsne()