from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import (
    DummySpecie,
    DummySpecies,
    Element,
    Specie,
    Species,
)


def generate_site_species_vector(structure: Structure, ATOM_NUM_UPPER: int) -> NDArray:
    if hasattr(structure, "species"):
        # atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = np.array(structure.atomic_numbers, dtype=np.int32)  # shape (num_sites,)
        x_species_vector = np.eye(ATOM_NUM_UPPER, dtype=np.float32)[atom_num - 1]  # shape (num_sites, ATOM_NUM_UPPER)

    else:
        x_species_vector = []

        for site in structure.species_and_occu:
            site_vector = np.zeros(ATOM_NUM_UPPER, dtype=np.float32)
            for elem in site.elements:
                if isinstance(elem, Element):
                    z = elem.Z
                    occ = site.element_composition[elem]
                elif isinstance(elem, (Specie, Species, Composition)):
                    z = elem.element.Z
                    occ = site.element_composition[elem.element]
                elif isinstance(elem, (DummySpecie, DummySpecies)):
                    raise ValueError(f"Unsupported specie: {site}! Skipped")
                else:
                    raise AttributeError(f"Unknown species type in site: {site}")

                if z <= 0 or z > ATOM_NUM_UPPER:
                    raise ValueError(f"Atomic number {z} out of bounds (1-{ATOM_NUM_UPPER})")

                site_vector[z - 1] += occ  # 0-based indexing

            x_species_vector.append(site_vector)

        x_species_vector = np.stack(x_species_vector, axis=0)
    if x_species_vector.ndim == 1:
        x_species_vector = x_species_vector[np.newaxis, :]
    return x_species_vector


class DistanceCalculator(Protocol):
    def calculate_pairwise(self, structure: Structure, edge_index: np.ndarray) -> np.ndarray: ...


class GaussianDistanceCalculator(DistanceCalculator):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(
        self,
        dmin: float = 0.0,
        dmax: float = 8.0,
        step: float = 0.2,
        var=None,
    ):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step, dtype=np.float32)
        if var is None:
            var = step
        self.var = var

    def calculate_pairwise(self, structure: Structure, edge_index: np.ndarray) -> NDArray:
        """
        Calculate pairwise Gaussian distance expansion. Use numpy to properly using multiprocessing

        Args:
        structure (Structure): A pymatgen Structure object.
        edge_index (torch.Tensor): A tensor of shape [2, num_edges] containing the edge indices.

        Returns:
        torch.Tensor: A tensor of shape [num_edges, num_gaussian_filters] containing the expanded distances.
        """
        assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        num_edges = edge_index.shape[1]

        # Calculate distances
        i_indices, j_indices = edge_index
        distances = np.array(
            [structure.get_distance(int(i), int(j)) for i, j in zip(i_indices, j_indices, strict=False)], dtype=np.float32
        ).reshape(-1, 1)  # Ensure shape is [num_edges, 1]

        # Compute Gaussian expansion
        return np.exp(-((distances - self.filter) ** 2) / (self.var**2))


def create_graph_data(
    data_input: dict[str, Any] | Structure,
    max_num_nbr: int,
    radius: float,
    calculators: list[DistanceCalculator],
) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
    structure = data_input if isinstance(data_input, Structure) else Structure.from_dict(data_input)

    node_features = generate_site_species_vector(structure, 96)

    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    num_sites = len(structure)
    total_edges = num_sites * max_num_nbr

    # preallocate
    edge_index_np = np.zeros((2, total_edges), dtype=np.int64)

    edge_count = 0

    for center, nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            # not find enough neighbors to build graph.
            # If it happens frequently, consider increase radius.
            nbr = nbr + [(structure[center], radius + 1.0, center)] * (max_num_nbr - len(nbr))
        else:
            nbr = nbr[:max_num_nbr]

        for i, neighbor in enumerate(nbr):
            edge_index_np[0, edge_count] = center
            edge_index_np[1, edge_count] = neighbor[2]
            edge_count += 1

    if len(calculators) == 1:
        edge_attr = calculators[0].calculate_pairwise(structure, edge_index_np)
    else:
        # Calculate edge attributes using all provided calculators
        # Numpy version not tested
        edge_attrs = []
        for calculator in calculators:
            edge_attr = calculator.calculate_pairwise(structure, edge_index_np)
            edge_attrs.append(edge_attr)

        # Concatenate all edge attributes
        edge_attr = np.concatenate(edge_attrs, axis=1)

    return node_features.tolist(), edge_index_np.tolist(), edge_attr.tolist()
