###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum, scatter_mean

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

import torch.nn as nn
import torch.nn.functional as F
import pickle
from copy import copy
with open(r'/home/u2023000178/lmz/mace/train_300K_svd_normalized.pkl', 'rb') as f:
    train_300K_svd_normalized = pickle.load(f)
    
with open(r'/home/u2023000178/lmz/mace/pos_index.pkl', 'rb') as f:
    pos_index = pickle.load(f)

with open(r'/home/u2023000178/lmz/mace/test_300K_svd_normalized.pkl', 'rb') as f:
    test_300K_svd_normalized = pickle.load(f)
    
with open(r'/home/u2023000178/lmz/mace/pos_index_test.pkl', 'rb') as f:
    pos_index_test = pickle.load(f)
# pylint: disable=C0302

# neighbor_acac = torch.tensor([ 1,  5,  6,  7, 11, 12, 13, 14,  0,  2,  3,  4,  5,  6,  7,  8,  9,
#        10, 11, 12, 13, 14,  1,  3,  4,  7,  8,  9, 10, 11,  1,  2,  4,  7,
#         8,  9, 10, 11,  1,  2,  3,  7,  8,  9, 10, 11,  0,  1,  6,  7, 11,
#        12, 13, 14,  0,  1,  5,  7, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,
#         6,  8,  9, 10, 11, 12, 13, 14,  1,  2,  3,  4,  7,  9, 10, 11,  1,
#         2,  3,  4,  7,  8, 10, 11,  1,  2,  3,  4,  7,  8,  9, 11,  0,  1,
#         2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14,  0,  1,  5,  6,  7,
#        11, 13, 14,  0,  1,  5,  6,  7, 12, 13, 14,  0,  1,  5,  6,  7, 11,
#        12, 13])

# index_acac = torch.tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
#         3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,
#         5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
#         7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,
#         9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
#        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
#        12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
#        14, 14])

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        # inter2 = interaction_cls_first(
        #     node_attrs_irreps=node_attr_irreps,
        #     node_feats_irreps=node_feats_irreps,
        #     edge_attrs_irreps=sh_irreps,
        #     edge_feats_irreps=edge_feats_irreps,
        #     target_irreps=interaction_irreps,
        #     hidden_irreps=hidden_irreps,
        #     avg_num_neighbors=avg_num_neighbors,
        #     radial_MLP=radial_MLP,
        # )
        # inter3 = interaction_cls_first(
        #     node_attrs_irreps=node_attr_irreps,
        #     node_feats_irreps=node_feats_irreps,
        #     edge_attrs_irreps=sh_irreps,
        #     edge_feats_irreps=edge_feats_irreps,
        #     target_irreps=interaction_irreps,
        #     hidden_irreps=hidden_irreps,
        #     avg_num_neighbors=avg_num_neighbors,
        #     radial_MLP=radial_MLP,
        # )
        self.interactions = torch.nn.ModuleList([inter])
        # self.interactions2 = torch.nn.ModuleList([inter2])
        # self.interactions3 = torch.nn.ModuleList([inter3])
        
        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        # prod2 = EquivariantProductBasisBlock(
        #     node_feats_irreps=node_feats_irreps_out,
        #     target_irreps=hidden_irreps,
        #     correlation=correlation[0],
        #     num_elements=num_elements,
        #     use_sc=use_sc_first,
        # )
        # prod3 = EquivariantProductBasisBlock(
        #     node_feats_irreps=node_feats_irreps_out,
        #     target_irreps=hidden_irreps,
        #     correlation=correlation[0],
        #     num_elements=num_elements,
        #     use_sc=use_sc_first,
        # )
        self.products = torch.nn.ModuleList([prod])
        # self.products2 = torch.nn.ModuleList([prod2])
        # self.products3 = torch.nn.ModuleList([prod3])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            # inter2 = interaction_cls(
            #     node_attrs_irreps=node_attr_irreps,
            #     node_feats_irreps=hidden_irreps,
            #     edge_attrs_irreps=sh_irreps,
            #     edge_feats_irreps=edge_feats_irreps,
            #     target_irreps=interaction_irreps,
            #     hidden_irreps=hidden_irreps_out,
            #     avg_num_neighbors=avg_num_neighbors,
            #     radial_MLP=radial_MLP,
            # )
            # inter3 = interaction_cls(
            #     node_attrs_irreps=node_attr_irreps,
            #     node_feats_irreps=hidden_irreps,
            #     edge_attrs_irreps=sh_irreps,
            #     edge_feats_irreps=edge_feats_irreps,
            #     target_irreps=interaction_irreps,
            #     hidden_irreps=hidden_irreps_out,
            #     avg_num_neighbors=avg_num_neighbors,
            #     radial_MLP=radial_MLP,
            # )
            self.interactions.append(inter)
            # self.interactions2.append(inter2)
            # self.interactions3.append(inter3)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            # prod2 = EquivariantProductBasisBlock(
            #     node_feats_irreps=interaction_irreps,
            #     target_irreps=hidden_irreps_out,
            #     correlation=correlation[i + 1],
            #     num_elements=num_elements,
            #     use_sc=True,
            # )
            # prod3 = EquivariantProductBasisBlock(
            #     node_feats_irreps=interaction_irreps,
            #     target_irreps=hidden_irreps_out,
            #     correlation=correlation[i + 1],
            #     num_elements=num_elements,
            #     use_sc=True,
            # )
            self.products.append(prod)
            # self.products2.append(prod2)
            # self.products3.append(prod3)
            
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))
        # from IPython import embed
        # embed()
        # exit(0)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        
        self.linear = nn.Sequential(  
            nn.Linear(24, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_2 = nn.Sequential(  
            nn.Linear(640 + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 640),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_3 = nn.Sequential(  
            nn.Linear(24, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_4 = nn.Sequential(  
            nn.Linear(1, 4),
            nn.SiLU(),
            nn.Linear(4, 8),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_5 = nn.Sequential(  
            nn.Linear(24, 128),
            nn.SiLU(),
            nn.Linear(128, 512),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_6 = nn.Sequential(  
            nn.Linear(128 + 128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.linear_7 = nn.Sequential(  
            nn.Linear(128 + 512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU()
            # nn.Linear(hidden_nf, message_nf),
            # self.act_fn,
        )
        
        self.tp = o3.FullTensorProduct("1o", "1o", ["1e"])

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        
        
        
        num_graphs = data["ptr"].numel() - 1
        
        # from IPython import embed
        # embed()
        device = data["positions"].device
    #     if data["node_attrs"].shape[0] == 75:
    #         index_acac = torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    #         1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
    #         3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,
    #         5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
    #         7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,
    #         9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
    #     11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
    #     12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
    #     14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    #     16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18,
    #     18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
    #     20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
    #     22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24,
    #     24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26,
    #     26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27,
    #     27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29,
    #     29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31,
    #     31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33,
    #     33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35,
    #     35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37,
    #     37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39,
    #     39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41,
    #     41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    #     42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44,
    #     44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    #     46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48,
    #     48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50,
    #     50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52,
    #     52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 54,
    #     54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56,
    #     56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57,
    #     57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59,
    #     59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    #     61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63,
    #     63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65,
    #     65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67,
    #     67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 69,
    #     69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71,
    #     71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72,
    #     72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74,
    #     74, 74]]).to(device)
    #         neighbor_acac = torch.tensor([[ 1,  5,  6,  7, 11, 12, 13, 14,  0,  2,  3,  4,  5,  6,  7,  8,  9,
    #             10, 11, 12, 13, 14,  1,  3,  4,  7,  8,  9, 10, 11,  1,  2,  4,  7,
    #                 8,  9, 10, 11,  1,  2,  3,  7,  8,  9, 10, 11,  0,  1,  6,  7, 11,
    #             12, 13, 14,  0,  1,  5,  7, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,
    #                 6,  8,  9, 10, 11, 12, 13, 14,  1,  2,  3,  4,  7,  9, 10, 11,  1,
    #                 2,  3,  4,  7,  8, 10, 11,  1,  2,  3,  4,  7,  8,  9, 11,  0,  1,
    #                 2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14,  0,  1,  5,  6,  7,
    #             11, 13, 14,  0,  1,  5,  6,  7, 11, 12, 14,  0,  1,  5,  6,  7, 11,
    #             12, 13, 16, 20, 21, 22, 26, 27, 28, 29, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    #         26, 27, 28, 29, 16, 18, 19, 22, 23, 24, 25, 26, 16, 17, 19, 22, 23, 24,
    #         25, 26, 16, 17, 18, 22, 23, 24, 25, 26, 15, 16, 21, 22, 26, 27, 28, 29,
    #         15, 16, 20, 22, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25,
    #         26, 27, 28, 29, 16, 17, 18, 19, 22, 24, 25, 26, 16, 17, 18, 19, 22, 23,
    #         25, 26, 16, 17, 18, 19, 22, 23, 24, 26, 15, 16, 17, 18, 19, 20, 21, 22,
    #         23, 24, 25, 27, 28, 29, 15, 16, 20, 21, 22, 26, 28, 29, 15, 16, 20, 21,
    #         22, 26, 27, 29, 15, 16, 20, 21, 22, 26, 27, 28, 31, 35, 36, 37, 41, 42, 43, 44, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    #         41, 42, 43, 44, 31, 33, 34, 37, 38, 39, 40, 41, 31, 32, 34, 37, 38, 39,
    #         40, 41, 31, 32, 33, 37, 38, 39, 40, 41, 30, 31, 36, 37, 41, 42, 43, 44,
    #         30, 31, 35, 37, 41, 42, 43, 44, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
    #         41, 42, 43, 44, 31, 32, 33, 34, 37, 39, 40, 41, 31, 32, 33, 34, 37, 38,
    #         40, 41, 31, 32, 33, 34, 37, 38, 39, 41, 30, 31, 32, 33, 34, 35, 36, 37,
    #         38, 39, 40, 42, 43, 44, 30, 31, 35, 36, 37, 41, 43, 44, 30, 31, 35, 36,
    #         37, 41, 42, 44, 30, 31, 35, 36, 37, 41, 42, 43, 46, 50, 51, 52, 56, 57, 58, 59, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    #         56, 57, 58, 59, 46, 48, 49, 52, 53, 54, 55, 56, 46, 47, 49, 52, 53, 54,
    #         55, 56, 46, 47, 48, 52, 53, 54, 55, 56, 45, 46, 51, 52, 56, 57, 58, 59,
    #         45, 46, 50, 52, 56, 57, 58, 59, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55,
    #         56, 57, 58, 59, 46, 47, 48, 49, 52, 54, 55, 56, 46, 47, 48, 49, 52, 53,
    #         55, 56, 46, 47, 48, 49, 52, 53, 54, 56, 45, 46, 47, 48, 49, 50, 51, 52,
    #         53, 54, 55, 57, 58, 59, 45, 46, 50, 51, 52, 56, 58, 59, 45, 46, 50, 51,
    #         52, 56, 57, 59, 45, 46, 50, 51, 52, 56, 57, 58, 61, 65, 66, 67, 71, 72, 73, 74, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    #         71, 72, 73, 74, 61, 63, 64, 67, 68, 69, 70, 71, 61, 62, 64, 67, 68, 69,
    #         70, 71, 61, 62, 63, 67, 68, 69, 70, 71, 60, 61, 66, 67, 71, 72, 73, 74,
    #         60, 61, 65, 67, 71, 72, 73, 74, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70,
    #         71, 72, 73, 74, 61, 62, 63, 64, 67, 69, 70, 71, 61, 62, 63, 64, 67, 68,
    #         70, 71, 61, 62, 63, 64, 67, 68, 69, 71, 60, 61, 62, 63, 64, 65, 66, 67,
    #         68, 69, 70, 72, 73, 74, 60, 61, 65, 66, 67, 71, 73, 74, 60, 61, 65, 66,
    #         67, 71, 72, 74, 60, 61, 65, 66, 67, 71, 72, 73]]).to(device)

    #         index_acac_2 = torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
    #     1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
    #     3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,
    #     5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    #     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,
    #     8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10,
    #    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    #    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13,
    #    13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16,
    #    16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
    #    18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20,
    #    20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    #    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23,
    #    23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25,
    #    25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
    #    26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28,
    #    28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31,
    #    31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33,
    #    33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35,
    #    35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36,
    #    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38,
    #    38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40,
    #    40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    #    41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43,
    #    43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46,
    #    46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48,
    #    48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50,
    #    50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51,
    #    52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53,
    #    53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55,
    #    55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    #    56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58,
    #    58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61,
    #    61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63,
    #    63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65,
    #    65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    #    67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68,
    #    68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70,
    #    70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71,
    #    71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73,
    #    73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74]]).to(device)
    #         neighbor_acac_2 = torch.tensor([[1,  2,  3,  4,  7,  8,  9, 10, 11,  0,  2,  3,  4,  5,  6,  7,  8,
    #     9, 10, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  0,  1,
    #     5,  6,  7, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  1,
    #     2,  3,  4,  7,  8,  9, 10, 11,  1,  2,  3,  4,  7,  8,  9, 10, 11,
    #     0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14,  0,  1,  5,
    #     6,  7, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  0,  1,
    #     5,  6,  7, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    #    10, 12, 13, 14,  1,  2,  3,  4,  7,  8,  9, 10, 11,  1,  2,  3,  4,
    #     7,  8,  9, 10, 11,  1,  2,  3,  4,  7,  8,  9, 10, 11, 16, 17, 18, 19, 22, 23, 24, 25, 26, 15, 17, 18, 19, 20, 21, 22, 23,
    #    24, 25, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 15, 16,
    #    20, 21, 22, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 16,
    #    17, 18, 19, 22, 23, 24, 25, 26, 16, 17, 18, 19, 22, 23, 24, 25, 26,
    #    15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 15, 16, 20,
    #    21, 22, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 15, 16,
    #    20, 21, 22, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    #    25, 27, 28, 29, 16, 17, 18, 19, 22, 23, 24, 25, 26, 16, 17, 18, 19,
    #    22, 23, 24, 25, 26, 16, 17, 18, 19, 22, 23, 24, 25, 26, 31, 32, 33, 34, 37, 38, 39, 40, 41, 30, 32, 33, 34, 35, 36, 37, 38,
    #    39, 40, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 30, 31,
    #    35, 36, 37, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 31,
    #    32, 33, 34, 37, 38, 39, 40, 41, 31, 32, 33, 34, 37, 38, 39, 40, 41,
    #    30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 30, 31, 35,
    #    36, 37, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 30, 31,
    #    35, 36, 37, 41, 42, 43, 44, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #    40, 42, 43, 44, 31, 32, 33, 34, 37, 38, 39, 40, 41, 31, 32, 33, 34,
    #    37, 38, 39, 40, 41, 31, 32, 33, 34, 37, 38, 39, 40, 41, 46, 47, 48, 49, 52, 53, 54, 55, 56, 45, 47, 48, 49, 50, 51, 52, 53,
    #    54, 55, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 45, 46,
    #    50, 51, 52, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 46,
    #    47, 48, 49, 52, 53, 54, 55, 56, 46, 47, 48, 49, 52, 53, 54, 55, 56,
    #    45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 45, 46, 50,
    #    51, 52, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 45, 46,
    #    50, 51, 52, 56, 57, 58, 59, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    #    55, 57, 58, 59, 46, 47, 48, 49, 52, 53, 54, 55, 56, 46, 47, 48, 49,
    #    52, 53, 54, 55, 56, 46, 47, 48, 49, 52, 53, 54, 55, 56, 61, 62, 63, 64, 67, 68, 69, 70, 71, 60, 62, 63, 64, 65, 66, 67, 68,
    #    69, 70, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 60, 61,
    #    65, 66, 67, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 61,
    #    62, 63, 64, 67, 68, 69, 70, 71, 61, 62, 63, 64, 67, 68, 69, 70, 71,
    #    60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 60, 61, 65,
    #    66, 67, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 60, 61,
    #    65, 66, 67, 71, 72, 73, 74, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    #    70, 72, 73, 74, 61, 62, 63, 64, 67, 68, 69, 70, 71, 61, 62, 63, 64,
    #    67, 68, 69, 70, 71, 61, 62, 63, 64, 67, 68, 69, 70, 71]]).to(device)
    #     elif data["node_attrs"].shape[0] == 150:
    #         index_acac = torch.tensor([[0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    #     1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
    #     3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,
    #     5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
    #     7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,
    #     9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
    #    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
    #    12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
    #    14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    #    16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18,
    #    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
    #    20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
    #    22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24,
    #    24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26,
    #    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27,
    #    27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29,
    #    29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31,
    #    31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33,
    #    33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35,
    #    35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37,
    #    37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39,
    #    39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41,
    #    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    #    42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44,
    #    44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    #    46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48,
    #    48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50,
    #    50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52,
    #    52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 54,
    #    54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56,
    #    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57,
    #    57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59,
    #    59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61,
    #    61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63,
    #    63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65,
    #    65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67,
    #    67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 69,
    #    69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71,
    #    71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72,
    #    72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74,
    #    74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76,
    #    76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78,
    #    78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80,
    #    80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82,
    #    82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 83, 83, 84,
    #    84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 86, 86,
    #    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87,
    #    87, 87, 87, 88, 88, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89,
    #    89, 89, 90,  90,  90,  90,  90,  90,  90,  90,  91,  91,  91,  91,  91,
    #     91,  91,  91,  91,  91,  91,  91,  91,  91,  92,  92,  92,  92,
    #     92,  92,  92,  92,  93,  93,  93,  93,  93,  93,  93,  93,  94,
    #     94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  95,  95,
    #     95,  95,  96,  96,  96,  96,  96,  96,  96,  96,  97,  97,  97,
    #     97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  98,  98,
    #     98,  98,  98,  98,  98,  98,  99,  99,  99,  99,  99,  99,  99,
    #     99, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101,
    #    101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102,
    #    102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103,
    #    104, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106,
    #    106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107,
    #    107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 109,
    #    109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110,
    #    110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112,
    #    112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113,
    #    113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114,
    #    114, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116,
    #    116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117,
    #    117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118,
    #    119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121,
    #    121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122,
    #    122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 124,
    #    124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125,
    #    125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127,
    #    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128, 128,
    #    128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129,
    #    129, 130, 130, 130, 130, 130, 130, 130, 130, 131, 131, 131, 131,
    #    131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 132, 132, 132,
    #    132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133, 133,
    #    134, 134, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 135, 135, 136, 136, 136, 136, 136,
    #    136, 136, 136, 136, 136, 136, 136, 136, 136, 137, 137, 137, 137,
    #    137, 137, 137, 137, 138, 138, 138, 138, 138, 138, 138, 138, 139,
    #    139, 139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 140, 140,
    #    140, 140, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142,
    #    142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 143, 143,
    #    143, 143, 143, 143, 143, 143, 144, 144, 144, 144, 144, 144, 144,
    #    144, 145, 145, 145, 145, 145, 145, 145, 145, 146, 146, 146, 146,
    #    146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 147, 147, 147,
    #    147, 147, 147, 147, 147, 148, 148, 148, 148, 148, 148, 148, 148,
    #    149, 149, 149, 149, 149, 149, 149, 149]]).to(device)
    #         neighbor_acac = torch.tensor([[1,  5,  6,  7, 11, 12, 13, 14,  0,  2,  3,  4,  5,  6,  7,  8,  9,
    #         10, 11, 12, 13, 14,  1,  3,  4,  7,  8,  9, 10, 11,  1,  2,  4,  7,
    #             8,  9, 10, 11,  1,  2,  3,  7,  8,  9, 10, 11,  0,  1,  6,  7, 11,
    #         12, 13, 14,  0,  1,  5,  7, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,
    #             6,  8,  9, 10, 11, 12, 13, 14,  1,  2,  3,  4,  7,  9, 10, 11,  1,
    #             2,  3,  4,  7,  8, 10, 11,  1,  2,  3,  4,  7,  8,  9, 11,  0,  1,
    #             2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14,  0,  1,  5,  6,  7,
    #         11, 13, 14,  0,  1,  5,  6,  7, 11, 12, 14,  0,  1,  5,  6,  7, 11,
    #         12, 13, 16, 20, 21, 22, 26, 27, 28, 29, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    #     26, 27, 28, 29, 16, 18, 19, 22, 23, 24, 25, 26, 16, 17, 19, 22, 23, 24,
    #     25, 26, 16, 17, 18, 22, 23, 24, 25, 26, 15, 16, 21, 22, 26, 27, 28, 29,
    #     15, 16, 20, 22, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25,
    #     26, 27, 28, 29, 16, 17, 18, 19, 22, 24, 25, 26, 16, 17, 18, 19, 22, 23,
    #     25, 26, 16, 17, 18, 19, 22, 23, 24, 26, 15, 16, 17, 18, 19, 20, 21, 22,
    #     23, 24, 25, 27, 28, 29, 15, 16, 20, 21, 22, 26, 28, 29, 15, 16, 20, 21,
    #     22, 26, 27, 29, 15, 16, 20, 21, 22, 26, 27, 28, 31, 35, 36, 37, 41, 42, 43, 44, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    #     41, 42, 43, 44, 31, 33, 34, 37, 38, 39, 40, 41, 31, 32, 34, 37, 38, 39,
    #     40, 41, 31, 32, 33, 37, 38, 39, 40, 41, 30, 31, 36, 37, 41, 42, 43, 44,
    #     30, 31, 35, 37, 41, 42, 43, 44, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
    #     41, 42, 43, 44, 31, 32, 33, 34, 37, 39, 40, 41, 31, 32, 33, 34, 37, 38,
    #     40, 41, 31, 32, 33, 34, 37, 38, 39, 41, 30, 31, 32, 33, 34, 35, 36, 37,
    #     38, 39, 40, 42, 43, 44, 30, 31, 35, 36, 37, 41, 43, 44, 30, 31, 35, 36,
    #     37, 41, 42, 44, 30, 31, 35, 36, 37, 41, 42, 43, 46, 50, 51, 52, 56, 57, 58, 59, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    #     56, 57, 58, 59, 46, 48, 49, 52, 53, 54, 55, 56, 46, 47, 49, 52, 53, 54,
    #     55, 56, 46, 47, 48, 52, 53, 54, 55, 56, 45, 46, 51, 52, 56, 57, 58, 59,
    #     45, 46, 50, 52, 56, 57, 58, 59, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55,
    #     56, 57, 58, 59, 46, 47, 48, 49, 52, 54, 55, 56, 46, 47, 48, 49, 52, 53,
    #     55, 56, 46, 47, 48, 49, 52, 53, 54, 56, 45, 46, 47, 48, 49, 50, 51, 52,
    #     53, 54, 55, 57, 58, 59, 45, 46, 50, 51, 52, 56, 58, 59, 45, 46, 50, 51,
    #     52, 56, 57, 59, 45, 46, 50, 51, 52, 56, 57, 58, 61, 65, 66, 67, 71, 72, 73, 74, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    #     71, 72, 73, 74, 61, 63, 64, 67, 68, 69, 70, 71, 61, 62, 64, 67, 68, 69,
    #     70, 71, 61, 62, 63, 67, 68, 69, 70, 71, 60, 61, 66, 67, 71, 72, 73, 74,
    #     60, 61, 65, 67, 71, 72, 73, 74, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70,
    #     71, 72, 73, 74, 61, 62, 63, 64, 67, 69, 70, 71, 61, 62, 63, 64, 67, 68,
    #     70, 71, 61, 62, 63, 64, 67, 68, 69, 71, 60, 61, 62, 63, 64, 65, 66, 67,
    #     68, 69, 70, 72, 73, 74, 60, 61, 65, 66, 67, 71, 73, 74, 60, 61, 65, 66,
    #     67, 71, 72, 74, 60, 61, 65, 66, 67, 71, 72, 73, 76,  80,  81,  82,  86,  87,  88,  89,  75,  77,  78,  79,  80,  81,
    #      82,  83,  84,  85,  86,  87,  88,  89,  76,  78,  79,  82,  83,  84,
    #      85,  86,  76,  77,  79,  82,  83,  84,  85,  86,  76,  77,  78,  82,
    #      83,  84,  85,  86,  75,  76,  81,  82,  86,  87,  88,  89,  75,  76,
    #      80,  82,  86,  87,  88,  89,  75,  76,  77,  78,  79,  80,  81,  83,
    #      84,  85,  86,  87,  88,  89,  76,  77,  78,  79,  82,  84,  85,  86,
    #      76,  77,  78,  79,  82,  83,  85,  86,  76,  77,  78,  79,  82,  83,
    #      84,  86,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  87,
    #      88,  89,  75,  76,  80,  81,  82,  86,  88,  89,  75,  76,  80,  81,
    #      82,  86,  87,  89,  75,  76,  80,  81,  82,  86,  87,  88,  91,  95,
    #      96,  97, 101, 102, 103, 104,  90,  92,  93,  94,  95,  96,  97,  98,
    #      99, 100, 101, 102, 103, 104,  91,  93,  94,  97,  98,  99, 100, 101,
    #      91,  92,  94,  97,  98,  99, 100, 101,  91,  92,  93,  97,  98,  99,
    #     100, 101,  90,  91,  96,  97, 101, 102, 103, 104,  90,  91,  95,  97,
    #     101, 102, 103, 104,  90,  91,  92,  93,  94,  95,  96,  98,  99, 100,
    #     101, 102, 103, 104,  91,  92,  93,  94,  97,  99, 100, 101,  91,  92,
    #      93,  94,  97,  98, 100, 101,  91,  92,  93,  94,  97,  98,  99, 101,
    #      90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 102, 103, 104,
    #      90,  91,  95,  96,  97, 101, 103, 104,  90,  91,  95,  96,  97, 101,
    #     102, 104,  90,  91,  95,  96,  97, 101, 102, 103, 106, 110, 111, 112,
    #     116, 117, 118, 119, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    #     116, 117, 118, 119, 106, 108, 109, 112, 113, 114, 115, 116, 106, 107,
    #     109, 112, 113, 114, 115, 116, 106, 107, 108, 112, 113, 114, 115, 116,
    #     105, 106, 111, 112, 116, 117, 118, 119, 105, 106, 110, 112, 116, 117,
    #     118, 119, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117,
    #     118, 119, 106, 107, 108, 109, 112, 114, 115, 116, 106, 107, 108, 109,
    #     112, 113, 115, 116, 106, 107, 108, 109, 112, 113, 114, 116, 105, 106,
    #     107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 105, 106,
    #     110, 111, 112, 116, 118, 119, 105, 106, 110, 111, 112, 116, 117, 119,
    #     105, 106, 110, 111, 112, 116, 117, 118, 121, 125, 126, 127, 131, 132,
    #     133, 134, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
    #     133, 134, 121, 123, 124, 127, 128, 129, 130, 131, 121, 122, 124, 127,
    #     128, 129, 130, 131, 121, 122, 123, 127, 128, 129, 130, 131, 120, 121,
    #     126, 127, 131, 132, 133, 134, 120, 121, 125, 127, 131, 132, 133, 134,
    #     120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134,
    #     121, 122, 123, 124, 127, 129, 130, 131, 121, 122, 123, 124, 127, 128,
    #     130, 131, 121, 122, 123, 124, 127, 128, 129, 131, 120, 121, 122, 123,
    #     124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 120, 121, 125, 126,
    #     127, 131, 133, 134, 120, 121, 125, 126, 127, 131, 132, 134, 120, 121,
    #     125, 126, 127, 131, 132, 133, 136, 140, 141, 142, 146, 147, 148, 149,
    #     135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    #     136, 138, 139, 142, 143, 144, 145, 146, 136, 137, 139, 142, 143, 144,
    #     145, 146, 136, 137, 138, 142, 143, 144, 145, 146, 135, 136, 141, 142,
    #     146, 147, 148, 149, 135, 136, 140, 142, 146, 147, 148, 149, 135, 136,
    #     137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 136, 137,
    #     138, 139, 142, 144, 145, 146, 136, 137, 138, 139, 142, 143, 145, 146,
    #     136, 137, 138, 139, 142, 143, 144, 146, 135, 136, 137, 138, 139, 140,
    #     141, 142, 143, 144, 145, 147, 148, 149, 135, 136, 140, 141, 142, 146,
    #     148, 149, 135, 136, 140, 141, 142, 146, 147, 149, 135, 136, 140, 141,
    #     142, 146, 147, 148]]).to(device)
    #         index_acac_2 = torch.tensor([[0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
    #     1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
    #     3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,
    #     5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    #     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,
    #     8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10,
    #    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    #    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13,
    #    13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
    #    15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    #    16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
    #    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20,
    #    20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    #    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23,
    #    23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25,
    #    25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
    #    26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28,
    #    28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30,
    #    30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
    #    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33,
    #    33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35,
    #    35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37,
    #    37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38,
    #    39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40,
    #    40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42,
    #    42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44,
    #    44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    #    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47,
    #    47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49,
    #    49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51,
    #    51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52,
    #    52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54,
    #    54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56,
    #    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57,
    #    57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59,
    #    59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61,
    #    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62,
    #    62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64,
    #    64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66,
    #    66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
    #    67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69,
    #    69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71,
    #    71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72,
    #    72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74,
    #    74, 74, 75,  75,  75,  75,  75,  75,  75,  75,  75,  76,  76,  76,  76,
    #     76,  76,  76,  76,  76,  76,  76,  76,  76,  76,  77,  77,  77,
    #     77,  77,  77,  77,  77,  77,  78,  78,  78,  78,  78,  78,  78,
    #     78,  78,  79,  79,  79,  79,  79,  79,  79,  79,  79,  80,  80,
    #     80,  80,  80,  80,  80,  80,  80,  81,  81,  81,  81,  81,  81,
    #     81,  81,  81,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,
    #     82,  82,  82,  82,  83,  83,  83,  83,  83,  83,  83,  83,  83,
    #     84,  84,  84,  84,  84,  84,  84,  84,  84,  85,  85,  85,  85,
    #     85,  85,  85,  85,  85,  86,  86,  86,  86,  86,  86,  86,  86,
    #     86,  86,  86,  86,  86,  86,  87,  87,  87,  87,  87,  87,  87,
    #     87,  87,  88,  88,  88,  88,  88,  88,  88,  88,  88,  89,  89,
    #     89,  89,  89,  89,  89,  89,  89,  90,  90,  90,  90,  90,  90,
    #     90,  90,  90,  91,  91,  91,  91,  91,  91,  91,  91,  91,  91,
    #     91,  91,  91,  91,  92,  92,  92,  92,  92,  92,  92,  92,  92,
    #     93,  93,  93,  93,  93,  93,  93,  93,  93,  94,  94,  94,  94,
    #     94,  94,  94,  94,  94,  95,  95,  95,  95,  95,  95,  95,  95,
    #     95,  96,  96,  96,  96,  96,  96,  96,  96,  96,  97,  97,  97,
    #     97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  98,  98,
    #     98,  98,  98,  98,  98,  98,  98,  99,  99,  99,  99,  99,  99,
    #     99,  99,  99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101,
    #    101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
    #    102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103,
    #    103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104,
    #    104, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106,
    #    106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107,
    #    107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108,
    #    108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110,
    #    110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111,
    #    111, 111, 111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112,
    #    112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113,
    #    113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115,
    #    115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116,
    #    116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117,
    #    117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118, 118, 119,
    #    119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120,
    #    120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 121,
    #    121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 122, 122,
    #    122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124,
    #    124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125,
    #    125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127,
    #    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128,
    #    128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129,
    #    129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130,
    #    131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
    #    131, 132, 132, 132, 132, 132, 132, 132, 132, 132, 133, 133, 133,
    #    133, 133, 133, 133, 133, 133, 134, 134, 134, 134, 134, 134, 134,
    #    134, 134, 135, 135, 135, 135, 135, 135, 135, 135, 135, 136, 136,
    #    136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 137,
    #    137, 137, 137, 137, 137, 137, 137, 137, 138, 138, 138, 138, 138,
    #    138, 138, 138, 138, 139, 139, 139, 139, 139, 139, 139, 139, 139,
    #    140, 140, 140, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141,
    #    141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 142, 142,
    #    142, 142, 142, 142, 142, 142, 143, 143, 143, 143, 143, 143, 143,
    #    143, 143, 144, 144, 144, 144, 144, 144, 144, 144, 144, 145, 145,
    #    145, 145, 145, 145, 145, 145, 145, 146, 146, 146, 146, 146, 146,
    #    146, 146, 146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147,
    #    147, 147, 147, 147, 148, 148, 148, 148, 148, 148, 148, 148, 148,
    #    149, 149, 149, 149, 149, 149, 149, 149, 149]]).to(device)
    #         neighbor_acac_2 = torch.tensor([[1,  2,  3,  4,  7,  8,  9, 10, 11,  0,  2,  3,  4,  5,  6,  7,  8,
    #     9, 10, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  0,  1,
    #     5,  6,  7, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  1,
    #     2,  3,  4,  7,  8,  9, 10, 11,  1,  2,  3,  4,  7,  8,  9, 10, 11,
    #     0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14,  0,  1,  5,
    #     6,  7, 11, 12, 13, 14,  0,  1,  5,  6,  7, 11, 12, 13, 14,  0,  1,
    #     5,  6,  7, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    #    10, 12, 13, 14,  1,  2,  3,  4,  7,  8,  9, 10, 11,  1,  2,  3,  4,
    #     7,  8,  9, 10, 11,  1,  2,  3,  4,  7,  8,  9, 10, 11, 16, 17, 18, 19, 22, 23, 24, 25, 26, 15, 17, 18, 19, 20, 21, 22, 23,
    #    24, 25, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 15, 16,
    #    20, 21, 22, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 16,
    #    17, 18, 19, 22, 23, 24, 25, 26, 16, 17, 18, 19, 22, 23, 24, 25, 26,
    #    15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 15, 16, 20,
    #    21, 22, 26, 27, 28, 29, 15, 16, 20, 21, 22, 26, 27, 28, 29, 15, 16,
    #    20, 21, 22, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    #    25, 27, 28, 29, 16, 17, 18, 19, 22, 23, 24, 25, 26, 16, 17, 18, 19,
    #    22, 23, 24, 25, 26, 16, 17, 18, 19, 22, 23, 24, 25, 26, 31, 32, 33, 34, 37, 38, 39, 40, 41, 30, 32, 33, 34, 35, 36, 37, 38,
    #    39, 40, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 30, 31,
    #    35, 36, 37, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 31,
    #    32, 33, 34, 37, 38, 39, 40, 41, 31, 32, 33, 34, 37, 38, 39, 40, 41,
    #    30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 30, 31, 35,
    #    36, 37, 41, 42, 43, 44, 30, 31, 35, 36, 37, 41, 42, 43, 44, 30, 31,
    #    35, 36, 37, 41, 42, 43, 44, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #    40, 42, 43, 44, 31, 32, 33, 34, 37, 38, 39, 40, 41, 31, 32, 33, 34,
    #    37, 38, 39, 40, 41, 31, 32, 33, 34, 37, 38, 39, 40, 41, 46, 47, 48, 49, 52, 53, 54, 55, 56, 45, 47, 48, 49, 50, 51, 52, 53,
    #    54, 55, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 45, 46,
    #    50, 51, 52, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 46,
    #    47, 48, 49, 52, 53, 54, 55, 56, 46, 47, 48, 49, 52, 53, 54, 55, 56,
    #    45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 45, 46, 50,
    #    51, 52, 56, 57, 58, 59, 45, 46, 50, 51, 52, 56, 57, 58, 59, 45, 46,
    #    50, 51, 52, 56, 57, 58, 59, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    #    55, 57, 58, 59, 46, 47, 48, 49, 52, 53, 54, 55, 56, 46, 47, 48, 49,
    #    52, 53, 54, 55, 56, 46, 47, 48, 49, 52, 53, 54, 55, 56, 61, 62, 63, 64, 67, 68, 69, 70, 71, 60, 62, 63, 64, 65, 66, 67, 68,
    #    69, 70, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 60, 61,
    #    65, 66, 67, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 61,
    #    62, 63, 64, 67, 68, 69, 70, 71, 61, 62, 63, 64, 67, 68, 69, 70, 71,
    #    60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 60, 61, 65,
    #    66, 67, 71, 72, 73, 74, 60, 61, 65, 66, 67, 71, 72, 73, 74, 60, 61,
    #    65, 66, 67, 71, 72, 73, 74, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    #    70, 72, 73, 74, 61, 62, 63, 64, 67, 68, 69, 70, 71, 61, 62, 63, 64,
    #    67, 68, 69, 70, 71, 61, 62, 63, 64, 67, 68, 69, 70, 71, 76,  77,  78,  79,  82,  83,  84,  85,  86,  75,  77,  78,  79,
    #     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  75,  76,  80,
    #     81,  82,  86,  87,  88,  89,  75,  76,  80,  81,  82,  86,  87,
    #     88,  89,  75,  76,  80,  81,  82,  86,  87,  88,  89,  76,  77,
    #     78,  79,  82,  83,  84,  85,  86,  76,  77,  78,  79,  82,  83,
    #     84,  85,  86,  75,  76,  77,  78,  79,  80,  81,  83,  84,  85,
    #     86,  87,  88,  89,  75,  76,  80,  81,  82,  86,  87,  88,  89,
    #     75,  76,  80,  81,  82,  86,  87,  88,  89,  75,  76,  80,  81,
    #     82,  86,  87,  88,  89,  75,  76,  77,  78,  79,  80,  81,  82,
    #     83,  84,  85,  87,  88,  89,  76,  77,  78,  79,  82,  83,  84,
    #     85,  86,  76,  77,  78,  79,  82,  83,  84,  85,  86,  76,  77,
    #     78,  79,  82,  83,  84,  85,  86,  91,  92,  93,  94,  97,  98,
    #     99, 100, 101,  90,  92,  93,  94,  95,  96,  97,  98,  99, 100,
    #    101, 102, 103, 104,  90,  91,  95,  96,  97, 101, 102, 103, 104,
    #     90,  91,  95,  96,  97, 101, 102, 103, 104,  90,  91,  95,  96,
    #     97, 101, 102, 103, 104,  91,  92,  93,  94,  97,  98,  99, 100,
    #    101,  91,  92,  93,  94,  97,  98,  99, 100, 101,  90,  91,  92,
    #     93,  94,  95,  96,  98,  99, 100, 101, 102, 103, 104,  90,  91,
    #     95,  96,  97, 101, 102, 103, 104,  90,  91,  95,  96,  97, 101,
    #    102, 103, 104,  90,  91,  95,  96,  97, 101, 102, 103, 104,  90,
    #     91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 102, 103, 104,
    #     91,  92,  93,  94,  97,  98,  99, 100, 101,  91,  92,  93,  94,
    #     97,  98,  99, 100, 101,  91,  92,  93,  94,  97,  98,  99, 100,
    #    101, 106, 107, 108, 109, 112, 113, 114, 115, 116, 105, 107, 108,
    #    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 105, 106,
    #    110, 111, 112, 116, 117, 118, 119, 105, 106, 110, 111, 112, 116,
    #    117, 118, 119, 105, 106, 110, 111, 112, 116, 117, 118, 119, 106,
    #    107, 108, 109, 112, 113, 114, 115, 116, 106, 107, 108, 109, 112,
    #    113, 114, 115, 116, 105, 106, 107, 108, 109, 110, 111, 113, 114,
    #    115, 116, 117, 118, 119, 105, 106, 110, 111, 112, 116, 117, 118,
    #    119, 105, 106, 110, 111, 112, 116, 117, 118, 119, 105, 106, 110,
    #    111, 112, 116, 117, 118, 119, 105, 106, 107, 108, 109, 110, 111,
    #    112, 113, 114, 115, 117, 118, 119, 106, 107, 108, 109, 112, 113,
    #    114, 115, 116, 106, 107, 108, 109, 112, 113, 114, 115, 116, 106,
    #    107, 108, 109, 112, 113, 114, 115, 116, 121, 122, 123, 124, 127,
    #    128, 129, 130, 131, 120, 122, 123, 124, 125, 126, 127, 128, 129,
    #    130, 131, 132, 133, 134, 120, 121, 125, 126, 127, 131, 132, 133,
    #    134, 120, 121, 125, 126, 127, 131, 132, 133, 134, 120, 121, 125,
    #    126, 127, 131, 132, 133, 134, 121, 122, 123, 124, 127, 128, 129,
    #    130, 131, 121, 122, 123, 124, 127, 128, 129, 130, 131, 120, 121,
    #    122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 120,
    #    121, 125, 126, 127, 131, 132, 133, 134, 120, 121, 125, 126, 127,
    #    131, 132, 133, 134, 120, 121, 125, 126, 127, 131, 132, 133, 134,
    #    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133,
    #    134, 121, 122, 123, 124, 127, 128, 129, 130, 131, 121, 122, 123,
    #    124, 127, 128, 129, 130, 131, 121, 122, 123, 124, 127, 128, 129,
    #    130, 131, 136, 137, 138, 139, 142, 143, 144, 145, 146, 135, 137,
    #    138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 135,
    #    136, 140, 141, 142, 146, 147, 148, 149, 135, 136, 140, 141, 142,
    #    146, 147, 148, 149, 135, 136, 140, 141, 142, 146, 147, 148, 149,
    #    136, 137, 138, 139, 142, 143, 144, 145, 146, 136, 137, 138, 139,
    #    142, 143, 144, 145, 146, 135, 136, 137, 138, 139, 140, 141, 143,
    #    144, 145, 146, 147, 148, 149, 135, 136, 140, 141, 142, 146, 147,
    #    148, 149, 135, 136, 140, 141, 142, 146, 147, 148, 149, 135, 136,
    #    140, 141, 142, 146, 147, 148, 149, 135, 136, 137, 138, 139, 140,
    #    141, 142, 143, 144, 145, 147, 148, 149, 136, 137, 138, 139, 142,
    #    143, 144, 145, 146, 136, 137, 138, 139, 142, 143, 144, 145, 146,
    #    136, 137, 138, 139, 142, 143, 144, 145, 146]]).to(device)
    #     # svd_embedding = torch.tensor([]).to(device)
        
        
        # index_1 = torch.linspace(3, 3 + (num_graphs - 1) * len(data["positions"]) / num_graphs, num_graphs).long()
        # pos_new = data["positions"][index_1, 1]
        # index_2 = torch.linspace(0, (num_graphs - 1) * len(data["positions"]) / num_graphs, num_graphs).long()
        # pos_new_0 = data["positions"][index_2, 0]
        # pos_new_1 = data["positions"][index_2, 1]
        # try:
        #     atom_pos_31 = [str(x).split(",")[0][7:] for x in pos_new]
        #     real_index = [pos_index[d] for d in atom_pos_31]
        #     temp = 1
        # except:
        #     atom_pos_00 = [str(x).split(",")[0][7:] for x in pos_new_0]
        #     atom_pos_01 = [str(x).split(",")[0][7:] for x in pos_new_1]
        #     real_index = [pos_index_test[d + e] for (d, e) in zip(atom_pos_00, atom_pos_01)]
        #     temp = 0
        
        # if temp:
        #     svd_batch = torch.stack([train_300K_svd_normalized[i] for i in real_index]).view(-1, 3).to(device)
        # else:
        #     svd_batch = torch.stack([test_300K_svd_normalized[i] for i in real_index]).view(-1, 3).to(device)
            
        # # hyper_edge_attrs = self.spherical_harmonics(svd_batch)
        # svd_batch_new = [x.view(-1, 1) for x in svd_batch.T]
        # svd_feat = [self.linear_4(x) for x in svd_batch_new]
        # # svd_feats = svd_feat[0]
        # # svd_feats = (svd_feat[0] + svd_feat[1] + svd_feat[2]) / 3
        # svd_feats = torch.cat([svd_feat[0], svd_feat[1], svd_feat[2]], dim=1)
        # # svd_feats = torch.cat([svd_batch_new[0], svd_batch_new[1], svd_batch_new[2]], dim=1)
        # # svd_embedding = self.linear(svd_feats)
        
        # if training:
        #     index = torch.linspace(3, 3 + (num_graphs - 1) * len(data["positions"]) / num_graphs, num_graphs).long()
        #     pos_new = data["positions"][index, 1]
        #     atom_pos_31 = [str(x).split(",")[0][7:] for x in pos_new]
        #     # from IPython import embed
        #     # embed()
        #     real_index = [pos_index[d] for d in atom_pos_31]
        #     svd_batch = torch.stack([train_300K_svd_normalized[i] for i in real_index]).view(-1, 3).to(device)
        #     svd_batch_new = [x.view(-1, 1) for x in svd_batch.T]
        #     svd_feat = [self.radial_embedding(x) for x in svd_batch_new]
        #     svd_feats = torch.cat([svd_feat[0], svd_feat[1], svd_feat[2]], dim=1)
        #     # svd_feats = torch.cat([svd_batch_new[0], svd_batch_new[1], svd_batch_new[2]], dim=1)
        #     svd_embedding = self.linear(svd_feats)
        # else:
        #     index = torch.linspace(0, (num_graphs - 1) * len(data["positions"]) / num_graphs, num_graphs).long()
        #     pos_new_0 = data["positions"][index, 0]
        #     pos_new_1 = data["positions"][index, 1]
        #     atom_pos_00 = [str(x).split(",")[0][7:] for x in pos_new_0]
        #     atom_pos_01 = [str(x).split(",")[0][7:] for x in pos_new_1]
        #     # from IPython import embed
        #     # embed()
        #     # exit(0)
        #     real_index = [pos_index_test[d + e] for (d, e) in zip(atom_pos_00, atom_pos_01)]
        #     svd_batch = torch.stack([test_300K_svd_normalized[i] for i in real_index]).view(-1, 3).to(device)
        #     svd_batch_new = [x.view(-1, 1) for x in svd_batch.T]
        #     svd_feat = [self.radial_embedding(x) for x in svd_batch_new]
        #     svd_feats = torch.cat([svd_feat[0], svd_feat[1], svd_feat[2]], dim=1)
        #     # svd_feats = torch.cat([svd_batch_new[0], svd_batch_new[1], svd_batch_new[2]], dim=1)
        #     svd_embedding = self.linear(svd_feats)
        # from IPython import embed
        # embed()
        
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # data["unit_shifts"]: [4532, 3]
        # data["edge_index"]: [2, 4532]
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies 
        # node_e0: [270]
        # e0: [10]
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]


        # Embeddings
        # node_feats: [270, 128]
        node_feats = self.node_embedding(data["node_attrs"])
        # vectors: [4532, 3]
        # lengths: [4532, 1]
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        
        
        # hyper_node_attrs = torch.stack([data["node_attrs"][x] for x in data["edge_index"][1]])
        # hyper_node_attrs = scatter_mean(src=hyper_node_attrs, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])
        
        
        
        # hyper_node_feats = torch.stack([node_feats[x] for x in data["edge_index"][1]])
        # hyper_node_feats = scatter_mean(src=hyper_node_feats, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])
        
        # # hyper_node_feats = 0.5 * hyper_node_feats + 0.5 * node_feats
        # # hyper_node_feats = node_feats
        
        # edge_index_acac = torch.cat([index_acac, neighbor_acac], dim=0)
        # edge_shifts_acac = torch.zeros([edge_index_acac.size()[1], 3]).to(device)
        # vectors_acac, lengths_acac = get_edge_vectors_and_lengths(
        #     positions=data["positions"],
        #     edge_index=edge_index_acac,
        #     shifts=edge_shifts_acac,
        # )
        # hyper_vectors_acac = scatter_mean(src=vectors_acac, index=edge_index_acac[0], dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths_acac = scatter_mean(src=lengths_acac, index=edge_index_acac[0], dim=0, dim_size=node_feats.shape[0])
        
        # hyper_edge_attrs_acac = self.spherical_harmonics(hyper_vectors_acac)
        # hyper_edge_feats_acac = self.radial_embedding(hyper_lengths_acac)
        
        # # if node_feats.shape[0] != 75 and node_feats.shape[0] != 150:
        # #     print(node_feats.shape[0])
        # # print(index_acac_2.shape)
        # # print(neighbor_acac_2.shape)
        # edge_index_acac_2 = torch.cat([index_acac_2, neighbor_acac_2], dim=0)
        # edge_shifts_acac_2 = torch.zeros([edge_index_acac_2.size()[1], 3]).to(device)
        # vectors_acac_2, lengths_acac_2 = get_edge_vectors_and_lengths(
        #     positions=data["positions"],
        #     edge_index=edge_index_acac_2,
        #     shifts=edge_shifts_acac_2,
        # )
        # hyper_vectors_acac_2 = scatter_mean(src=vectors_acac_2, index=edge_index_acac_2[0], dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths_acac_2 = scatter_mean(src=lengths_acac_2, index=edge_index_acac_2[0], dim=0, dim_size=node_feats.shape[0])
        
        # hyper_edge_attrs_acac_2 = self.spherical_harmonics(hyper_vectors_acac_2)
        # hyper_edge_feats_acac_2 = self.radial_embedding(hyper_lengths_acac_2)
        

        # if node_feats.shape[0] != 75 and node_feats.shape[0] != 150:
        #     print(node_feats.shape[0])


        # neighbor_3bpa_7 = torch.tensor([[3, 2, 1, 12, 10, 5, 4],[5, 0, 4, 3, 13, 7, 2],[12, 10, 0, 11, 3, 7, 1],
        #                                 [0, 1, 2, 5, 12, 20, 4],[7, 6, 1, 8, 9, 10, 0],[1, 13, 15, 16, 3, 0, 14],
        #                                 [8, 9, 4, 7, 1, 15, 5],[10, 4, 11, 2, 8, 1, 6],[6, 9, 4, 7, 1, 15, 10],
        #                                 [6, 8, 4, 15, 1, 5, 7],[11, 7, 2, 12, 4, 0, 1],[10, 7, 2, 12, 4, 0, 1],
        #                                 [2, 10, 0, 11, 3, 7, 1],[15, 16, 5, 14, 1, 17, 18],[17, 18, 13, 20, 16, 26, 15],
        #                                 [13, 16, 5, 14, 9, 1, 6],[13, 15, 14, 5, 1, 20, 17],[20, 19, 14, 22, 18, 21, 13],
        #                                 [26, 14, 23, 25, 17, 21, 13],[22, 17, 21, 24, 20, 14, 23],[17, 19, 14, 22, 13, 5, 16],
        #                                 [24, 23, 19, 25, 22, 17, 18],[19, 17, 21, 24, 20, 14, 23],[25, 21, 18, 26, 24, 19, 14],
        #                                 [21, 19, 23, 22, 25, 17, 18],[23, 21, 18, 24, 26, 19, 14],[18, 23, 14, 25, 13, 15, 21]]).to(device)
        # index_3bpa_7 = torch.tensor([[0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2, 2],
        #                              [3, 3, 3, 3, 3, 3, 3],[4, 4, 4, 4, 4, 4, 4],[5, 5, 5, 5, 5, 5, 5],
        #                              [6, 6, 6, 6, 6, 6, 6],[7, 7, 7, 7, 7, 7, 7],[8, 8, 8, 8, 8, 8, 8],
        #                              [9, 9, 9, 9, 9, 9, 9],[10, 10, 10, 10, 10, 10, 10],[11, 11, 11, 11, 11, 11, 11],
        #                              [12, 12, 12, 12, 12, 12, 12],[13, 13, 13, 13, 13, 13, 13],[14, 14, 14, 14, 14, 14, 14],
        #                              [15, 15, 15, 15, 15, 15, 15],[16, 16, 16, 16, 16, 16, 16],[17, 17, 17, 17, 17, 17, 17],
        #                              [18, 18, 18, 18, 18, 18, 18],[19, 19, 19, 19, 19, 19, 19],[20, 20, 20, 20, 20, 20, 20],
        #                              [21, 21, 21, 21, 21, 21, 21],[22, 22, 22, 22, 22, 22, 22],[23, 23, 23, 23, 23, 23, 23],
        #                              [24, 24, 24, 24, 24, 24, 24],[25, 25, 25, 25, 25, 25, 25],[26, 26, 26, 26, 26, 26, 26]]).to(device)
        
        # neighbor_3bpa_10 = torch.tensor([[3, 2, 1, 12, 10, 5, 4, 7, 11, 16],[5, 0, 4, 3, 13, 7, 2, 16, 6, 15],[12, 10, 0, 11, 3, 7, 1, 4, 5, 6],
        #                                  [0, 1, 2, 5, 12, 20, 4, 10, 13, 16],[7, 6, 1, 8, 9, 10, 0, 5, 2, 15],[1, 13, 15, 16, 3, 0, 14, 4, 9, 20],
        #                                  [8, 9, 4, 7, 1, 15, 5, 13, 10, 16],[10, 4, 11, 2, 8, 1, 6, 0, 9, 12],[6, 9, 4, 7, 1, 15, 10, 5, 11, 0],
        #                                  [6, 8, 4, 15, 1, 5, 7, 13, 0, 16],[11, 7, 2, 12, 4, 0, 1, 3, 6, 8],[10, 7, 2, 12, 4, 0, 1, 8, 6, 3],
        #                                  [2, 10, 0, 11, 3, 7, 1, 4, 5, 16],[15, 16, 5, 14, 1, 17, 18, 20, 26, 4],[17, 18, 13, 20, 16, 26, 15, 19, 5, 23],
        #                                  [13, 16, 5, 14, 9, 1, 6, 26, 4, 18],[13, 15, 14, 5, 1, 20, 17, 18, 4, 26],[20, 19, 14, 22, 18, 21, 13, 23, 16, 5],
        #                                  [26, 14, 23, 25, 17, 21, 13, 19, 15, 16],[22, 17, 21, 24, 20, 14, 23, 18, 25, 13],[17, 19, 14, 22, 13, 5, 16, 3, 21, 18],
        #                                  [24, 23, 19, 25, 22, 17, 18, 14, 20, 26],[19, 17, 21, 24, 20, 14, 23, 18, 25, 13],[25, 21, 18, 26, 24, 19, 14, 17, 22, 20],
        #                                  [21, 19, 23, 22, 25, 17, 18, 14, 20, 26],[23, 21, 18, 24, 26, 19, 14, 17, 22, 20],[18, 23, 14, 25, 13, 15, 21, 17, 16, 19]]).to(device)
        # index_3bpa_10 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #                               [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        #                               [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],[7, 7, 7, 7, 7, 7, 7, 7, 7, 7],[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        #                               [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],[11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        #                               [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],[13, 13, 13, 13, 13, 13, 13, 13, 13, 13],[14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        #                               [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],[16, 16, 16, 16, 16, 16, 16, 16, 16, 16],[17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
        #                               [18, 18, 18, 18, 18, 18, 18, 18, 18, 18],[19, 19, 19, 19, 19, 19, 19, 19, 19, 19],[20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        #                               [21, 21, 21, 21, 21, 21, 21, 21, 21, 21],[22, 22, 22, 22, 22, 22, 22, 22, 22, 22],[23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
        #                               [24, 24, 24, 24, 24, 24, 24, 24, 24, 24],[25, 25, 25, 25, 25, 25, 25, 25, 25, 25],[26, 26, 26, 26, 26, 26, 26, 26, 26, 26]]).to(device)
        
        # neighbor_acac_6 = torch.tensor([[5, 1, 6, 14, 7, 12],[7, 2, 0, 11, 3, 5],[3, 1, 4, 11, 7, 8],
        #                                 [11, 2, 4, 1, 10, 5],[8, 10, 9, 2, 3, 1],[0, 11, 1, 6, 3, 12],
        #                                 [12, 13, 14, 0, 5, 1],[1, 2, 0, 9, 13, 4],[4, 9, 10, 2, 7, 1],
        #                                 [4, 8, 10, 2, 7, 1],[4, 8, 9, 2, 3, 11],[3, 5, 2, 0, 1, 4],
        #                                 [6, 13, 14, 0, 5, 1],[6, 12, 14, 0, 7, 1],[6, 13, 12, 0, 5, 1]]).to(device)
        
        # index_acac_6 = torch.tensor([[0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2],
        #                              [3, 3, 3, 3, 3, 3],[4, 4, 4, 4, 4, 4],[5, 5, 5, 5, 5, 5],
        #                              [6, 6, 6, 6, 6, 6],[7, 7, 7, 7, 7, 7],[8, 8, 8, 8, 8, 8],
        #                              [9, 9, 9, 9, 9, 9],[10, 10, 10, 10, 10, 10],[11, 11, 11, 11, 11, 11],
        #                              [12, 12, 12, 12, 12, 12],[13, 13, 13, 13, 13, 13],[14, 14, 14, 14, 14, 14]]).to(device)
        
        # neighbor_acac_5 = torch.tensor([[5, 1, 6, 14, 7],[7, 2, 0, 11, 3],[3, 1, 4, 11, 7],
        #                                 [11, 2, 4, 1, 10],[8, 10, 9, 2, 3],[0, 11, 1, 6, 3],
        #                                 [12, 13, 14, 0, 5],[1, 2, 0, 9, 13],[4, 9, 10, 2, 7],
        #                                 [4, 8, 10, 2, 7],[4, 8, 9, 2, 3],[3, 5, 2, 0, 1],
        #                                 [6, 13, 14, 0, 5],[6, 12, 14, 0, 7],[6, 13, 12, 0, 5]]).to(device)
        
        # index_acac_5 = torch.tensor([[0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[2, 2, 2, 2, 2],
        #                              [3, 3, 3, 3, 3],[4, 4, 4, 4, 4],[5, 5, 5, 5, 5],
        #                              [6, 6, 6, 6, 6],[7, 7, 7, 7, 7],[8, 8, 8, 8, 8],
        #                              [9, 9, 9, 9, 9],[10, 10, 10, 10, 10],[11, 11, 11, 11, 11],
        #                              [12, 12, 12, 12, 12],[13, 13, 13, 13, 13],[14, 14, 14, 14, 14]]).to(device)
        
        # time = int(node_feats.shape[0] / neighbor_acac_5.shape[0])
        # num = index_acac_5.shape[0]
        # index2 = copy(index_acac_5)
        # neighbor_2 = copy(neighbor_acac_5)
        # for i in range(time - 1):
        #     index2 = index2 + num
        #     neighbor_2 = neighbor_2 + num
        #     index_acac_5 = torch.cat([index_acac_5, index2], dim=0)
        #     neighbor_acac_5 = torch.cat([neighbor_acac_5, neighbor_2], dim=0)
        # index_acac_5 = index_acac_5.reshape(-1)
        # neighbor_acac_5 = neighbor_acac_5.reshape(-1)
        # hyper_vectors = scatter_mean(src=vectors[neighbor_acac_5], index=index_acac_5, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths = scatter_mean(src=lengths[neighbor_acac_5], index=index_acac_5, dim=0, dim_size=node_feats.shape[0])
        
        
        # time = int(node_feats.shape[0] / neighbor_acac_6.shape[0])
        # num = index_acac_6.shape[0]
        # index2 = copy(index_acac_6)
        # neighbor_2 = copy(neighbor_acac_6)
        # for i in range(time - 1):
        #     index2 = index2 + num
        #     neighbor_2 = neighbor_2 + num
        #     index_acac_6 = torch.cat([index_acac_6, index2], dim=0)
        #     neighbor_acac_6 = torch.cat([neighbor_acac_6, neighbor_2], dim=0)
        # index_acac_6 = index_acac_6.reshape(-1)
        # neighbor_acac_6 = neighbor_acac_6.reshape(-1)
        # hyper_vectors = scatter_mean(src=vectors[neighbor_acac_6], index=index_acac_6, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths = scatter_mean(src=lengths[neighbor_acac_6], index=index_acac_6, dim=0, dim_size=node_feats.shape[0])
        
        # time = int(node_feats.shape[0] / neighbor_3bpa_10.shape[0])
        # num = index_3bpa_10.shape[0]
        # index2 = copy(index_3bpa_10)
        # neighbor_2 = copy(neighbor_3bpa_10)
        # for i in range(time - 1):
        #     index2 = index2 + num
        #     neighbor_2 = neighbor_2 + num
        #     index_3bpa_10 = torch.cat([index_3bpa_10, index2], dim=0)
        #     neighbor_3bpa_10 = torch.cat([neighbor_3bpa_10, neighbor_2], dim=0)
            

        # index_3bpa_10 = index_3bpa_10.reshape(-1)
        # neighbor_3bpa_10 = neighbor_3bpa_10.reshape(-1)
        
        # hyper_vectors = scatter_mean(src=vectors[neighbor_3bpa_10], index=index_3bpa_10, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths = scatter_mean(src=lengths[neighbor_3bpa_10], index=index_3bpa_10, dim=0, dim_size=node_feats.shape[0])
        
        # time = int(node_feats.shape[0] / neighbor_3bpa_7.shape[0])
        # num = index_3bpa_7.shape[0]
        # index2 = copy(index_3bpa_7)
        # neighbor_2 = copy(neighbor_3bpa_7)
        # for i in range(time - 1):
        #     index2 = index2 + num
        #     neighbor_2 = neighbor_2 + num
        #     index_3bpa_7 = torch.cat([index_3bpa_7, index2], dim=0)
        #     neighbor_3bpa_7 = torch.cat([neighbor_3bpa_7, neighbor_2], dim=0)
        # index_3bpa_7 = index_3bpa_7.reshape(-1)
        # neighbor_3bpa_7 = neighbor_3bpa_7.reshape(-1)
        # hyper_vectors = scatter_mean(src=vectors[neighbor_3bpa_7], index=index_3bpa_7, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths = scatter_mean(src=lengths[neighbor_3bpa_7], index=index_3bpa_7, dim=0, dim_size=node_feats.shape[0])
        
        # from IPython import embed
        # embed()
        # exit(0)
        

        
        # hyper_vectors = scatter_mean(src=vectors, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths = scatter_mean(src=lengths, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])
        
        # hyper_edge_attrs = self.spherical_harmonics(hyper_vectors)
        # hyper_edge_feats = self.radial_embedding(hyper_lengths)
        
        # hyper_index = torch.arange(0, node_feats.shape[0]).to(device)
        # hyper_index = torch.cat([hyper_index, hyper_index]).view(2, -1)
        
        
        index_geq = (lengths <= 4).reshape(-1) & (lengths >= 2).reshape(-1)
        index_leq = (lengths <= 3).reshape(-1)
        hyper_edge_index_11 = data["edge_index"][0][index_geq]
        hyper_edge_index_12 = data["edge_index"][1][index_geq]
        hyper_edge_index_21 = data["edge_index"][0][index_leq]
        hyper_edge_index_22 = data["edge_index"][1][index_leq]
        
        # hyper_index_1 = torch.tensor([]).to(device)
        # hyper_index_2 = torch.tensor([]).to(device)
        # # hyper_index_3 = torch.tensor([]).to(device)
        # # hyper_index_4 = torch.tensor([]).to(device)

        # for i in range(node_feats.shape[0]):
        #     ind = torch.where(data["edge_index"][0] == i)[0]
        #     hyper_index_1 = torch.cat([hyper_index_1, ind[:int(len(ind) / 2)]])
        #     hyper_index_2 = torch.cat([hyper_index_2, ind[int(len(ind) / 2):]])
            
        # # for i in range(node_feats.shape[0]):
        # #     ind = torch.where(data["edge_index"][0] == i)[0]
        # #     hyper_index_1 = torch.cat([hyper_index_1, ind[:int(len(ind) / 3)]])
        # #     hyper_index_2 = torch.cat([hyper_index_2, ind[int(len(ind) / 3): int(len(ind) / 3 * 2)]])
        # #     hyper_index_3 = torch.cat([hyper_index_3, ind[int(len(ind) / 3 * 2):]])
        
        # # for i in range(node_feats.shape[0]):
        # #     ind = torch.where(data["edge_index"][0] == i)[0]
        # #     hyper_index_1 = torch.cat([hyper_index_1, ind[:int(len(ind) / 4)]])
        # #     hyper_index_2 = torch.cat([hyper_index_2, ind[int(len(ind) / 4): int(len(ind) / 4 * 2)]])
        # #     hyper_index_3 = torch.cat([hyper_index_3, ind[int(len(ind) / 4 * 2): int(len(ind) / 4 * 3)]])
        # #     hyper_index_4 = torch.cat([hyper_index_3, ind[int(len(ind) / 4 * 3):]])
            
        # hyper_edge_index_11 = data["edge_index"][0][hyper_index_1.long()]
        # hyper_edge_index_12 = data["edge_index"][1][hyper_index_1.long()]
        # hyper_edge_index_21 = data["edge_index"][0][hyper_index_2.long()]
        # hyper_edge_index_22 = data["edge_index"][1][hyper_index_2.long()]
        # # hyper_edge_index_31 = data["edge_index"][0][hyper_index_3.long()]
        # # hyper_edge_index_32 = data["edge_index"][1][hyper_index_3.long()]
        # # hyper_edge_index_41 = data["edge_index"][0][hyper_index_4.long()]
        # # hyper_edge_index_42 = data["edge_index"][1][hyper_index_4.long()]
        
        hyper_vectors_1 = scatter_mean(src=vectors[hyper_edge_index_12], index=hyper_edge_index_11, dim=0, dim_size=node_feats.shape[0])
        hyper_vectors_2 = scatter_mean(src=vectors[hyper_edge_index_22], index=hyper_edge_index_21, dim=0, dim_size=node_feats.shape[0])
        # hyper_vectors_3 = scatter_mean(src=vectors[hyper_edge_index_32], index=hyper_edge_index_31, dim=0, dim_size=node_feats.shape[0])
        # hyper_vectors_4 = scatter_mean(src=vectors[hyper_edge_index_42], index=hyper_edge_index_41, dim=0, dim_size=node_feats.shape[0])
        
        hyper_lengths_1 = scatter_mean(src=lengths[hyper_edge_index_12], index=hyper_edge_index_11, dim=0, dim_size=node_feats.shape[0])
        hyper_lengths_2 = scatter_mean(src=lengths[hyper_edge_index_22], index=hyper_edge_index_21, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths_3 = scatter_mean(src=lengths[hyper_edge_index_32], index=hyper_edge_index_31, dim=0, dim_size=node_feats.shape[0])
        # hyper_lengths_4 = scatter_mean(src=lengths[hyper_edge_index_42], index=hyper_edge_index_41, dim=0, dim_size=node_feats.shape[0])
        
        hyper_lengths_squre = torch.sqrt(scatter_mean(src=lengths * lengths, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0]))
        hyper_lengths_1_squre = torch.sqrt(scatter_mean(src=lengths[hyper_edge_index_12] * lengths[hyper_edge_index_12], index=hyper_edge_index_11, dim=0, dim_size=node_feats.shape[0]))
        hyper_lengths_2_squre = torch.sqrt(scatter_mean(src=lengths[hyper_edge_index_22] * lengths[hyper_edge_index_22], index=hyper_edge_index_21, dim=0, dim_size=node_feats.shape[0]))
        
        
        # hyper_lengths_cubic = (scatter_mean(src=lengths * lengths * lengths, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])) ** (1 / 3)
        # hyper_lengths_1_cubic = (scatter_mean(src=lengths[hyper_edge_index_12] * lengths[hyper_edge_index_12] * lengths[hyper_edge_index_12], index=hyper_edge_index_11, dim=0, dim_size=node_feats.shape[0])) ** (1 / 3)
        # hyper_lengths_2_cubic = (scatter_mean(src=lengths[hyper_edge_index_22] * lengths[hyper_edge_index_22] * lengths[hyper_edge_index_22], index=hyper_edge_index_21, dim=0, dim_size=node_feats.shape[0])) ** (1 / 3)
        
        # from IPython import embed
        # embed()
        # exit(0)
        
        # vectors_squre = vectors * vectors
        # lengths_squre = lengths * lengths
        # edge_attrs: [4532, 16]
        # edge_feats: [4532, 8]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        # edge_feats = (self.radial_embedding(lengths) + self.radial_embedding(lengths_squre)) / 2

        # hyper_vectors_neighbor_1 = data["positions"] - hyper_vectors_1
        # hyper_vectors_neighbor_2 = data["positions"] - hyper_vectors_2
        
        # hyper_vectors_prod_1 = self.tp(data["positions"], hyper_vectors_neighbor_1)
        # hyper_vectors_prod_2 = self.tp(data["positions"], hyper_vectors_neighbor_2)
        
        # hyper_edge_attrs_prod_1 = self.spherical_harmonics(hyper_vectors_prod_1)
        # hyper_edge_attrs_prod_2 = self.spherical_harmonics(hyper_vectors_prod_2)
        
        hyper_edge_attrs_1 = self.spherical_harmonics(hyper_vectors_1)
        hyper_edge_attrs_2 = self.spherical_harmonics(hyper_vectors_2)
        # hyper_edge_attrs_3 = self.spherical_harmonics(hyper_vectors_3)
        # hyper_edge_attrs_4 = self.spherical_harmonics(hyper_vectors_4)
        
        hyper_edge_feats_1 = self.radial_embedding(hyper_lengths_1)
        hyper_edge_feats_2 = self.radial_embedding(hyper_lengths_2)
        # hyper_edge_feats_3 = self.radial_embedding(hyper_lengths_3)
        # hyper_edge_feats_4 = self.radial_embedding(hyper_lengths_4)
        
        hyper_edge_feats_squre = self.radial_embedding(hyper_lengths_squre)
        hyper_edge_feats_1_squre = self.radial_embedding(hyper_lengths_1_squre)
        hyper_edge_feats_2_squre = self.radial_embedding(hyper_lengths_2_squre)
        
        # hyper_edge_feats_cubic = self.radial_embedding(hyper_lengths_cubic)
        # hyper_edge_feats_1_cubic = self.radial_embedding(hyper_lengths_1_cubic)
        # hyper_edge_feats_2_cubic = self.radial_embedding(hyper_lengths_2_cubic)
        
        hyper_edge_attrs_1 = self.spherical_harmonics(hyper_vectors_1)
        hyper_edge_attrs_2 = self.spherical_harmonics(hyper_vectors_2)
        # hyper_edge_attrs_3 = self.spherical_harmonics(hyper_vectors_3)
        # hyper_edge_attrs_4 = self.spherical_harmonics(hyper_vectors_4)
        
        # hyper_edge_feats_1 = self.radial_embedding((hyper_lengths_1 + hyper_lengths_1_squre) / 2)
        # hyper_edge_feats_2 = self.radial_embedding((hyper_lengths_2 + hyper_lengths_2_squre) / 2)
        # hyper_edge_feats_1 = (self.radial_embedding(hyper_lengths_1) + self.radial_embedding(hyper_lengths_1_squre) + self.radial_embedding(hyper_lengths_1_cubic)) / 3
        # hyper_edge_feats_2 = (self.radial_embedding(hyper_lengths_2) + self.radial_embedding(hyper_lengths_2_squre) + self.radial_embedding(hyper_lengths_2_cubic)) / 3
        hyper_edge_feats_1 = (self.radial_embedding(hyper_lengths_1) + self.radial_embedding(hyper_lengths_1_squre)) / 2
        hyper_edge_feats_2 = (self.radial_embedding(hyper_lengths_2) + self.radial_embedding(hyper_lengths_2_squre)) / 2
        # hyper_edge_feats_1 = self.radial_embedding(hyper_lengths_1)
        # hyper_edge_feats_2 = self.radial_embedding(hyper_lengths_2)
        # hyper_edge_feats_3 = self.radial_embedding(hyper_lengths_3)
        # hyper_edge_feats_4 = self.radial_embedding(hyper_lengths_4)
        
        # hyper_edge_attrs = self.spherical_harmonics(hyper_vectors)
        # hyper_edge_feats = (self.radial_embedding(hyper_lengths) + self.radial_embedding(hyper_lengths_squre)) / 2
        # # hyper_edge_feats = (self.radial_embedding(hyper_lengths) + self.radial_embedding(hyper_lengths_squre) + self.radial_embedding(hyper_lengths_cubic)) / 3
        # Interactions
        node_es_list = []
        node_feats_list = []
        # from IPython import embed
        # embed()
        # exit(0)
        
        # for interaction3, product3, readout in zip(
        #     self.interactions, self.products, self.readouts
        # ):
        #     node_feats, sc = interaction3(
        #         node_attrs=data["node_attrs"],
        #         node_feats=node_feats,
        #         edge_attrs=hyper_edge_attrs,
        #         edge_feats=hyper_edge_feats,
        #         edge_index=hyper_index,
        #         svd_feats=svd_feats,
        #         training=training,
        #     )
            
        #     # node_feats = node_feats + svd_embedding.unsqueeze(-1)
        #     # from IPython import embed
        #     # embed()
        #     # exit(0)
        #     # node_feats: [270, 512]、[270, 128]
        #     node_feats = product3(
        #         node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
        #     )
        #     # node_feats_list.append(node_feats)
        #     # node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # for interaction, product, readout in zip(
        #     self.interactions, self.products, self.readouts
        # ):
        #     # from IPython import embed
        #     # embed()
        #     # node_feats: [270, 128, 16]
        #     # edge_attrs: [4532, 16]
        #     # edge_feats: [4532, 8]
            
        #     node_feats, sc = interaction(
        #         node_attrs=data["node_attrs"],
        #         node_feats=node_feats,
        #         edge_attrs=edge_attrs,
        #         edge_feats=edge_feats,
        #         edge_index=data["edge_index"],
        #         svd_feats=svd_embedding,
        #         training=training,
        #     )
            
        #     # node_feats = node_feats + svd_embedding.unsqueeze(-1)
        #     # from IPython import embed
        #     # embed()
        #     # exit(0)
        #     # node_feats: [270, 512]、[270, 128]
        #     node_feats = product(
        #         node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
        #     )
        #     node_feats_list.append(node_feats)
        #     node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
            
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            # hyper_node_feats = torch.stack([node_feats[x] for x in data["edge_index"][1]])
            # hyper_node_feats = scatter_mean(src=hyper_node_feats, index=data["edge_index"][0], dim=0, dim_size=node_feats.shape[0])
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                # svd_feats=torch.tensor([]).to(device),
                # svd_attrs=torch.tensor([]).to(device),
                # svd_feats=hyper_edge_feats,
                # svd_attrs=hyper_edge_attrs,
                # svd_feats=[hyper_edge_feats_1, hyper_edge_feats_2],
                # svd_attrs=[hyper_edge_attrs_prod_1, hyper_edge_attrs_prod_2],
                # svd_feats=[hyper_edge_feats_1, hyper_edge_feats_2, hyper_edge_feats_1_squre, hyper_edge_feats_2_squre],
                # svd_attrs=[hyper_edge_attrs_1, hyper_edge_attrs_2],
                # svd_feats=[hyper_edge_feats, hyper_edge_feats_1, hyper_edge_feats_2],
                # svd_attrs=[hyper_edge_attrs, hyper_edge_attrs_1, hyper_edge_attrs_2],
                # svd_feats=[hyper_edge_feats_acac, hyper_edge_feats_acac_2],
                # svd_attrs=[hyper_edge_attrs_acac, hyper_edge_attrs_acac_2],
                svd_feats=hyper_edge_feats_2,
                svd_attrs=hyper_edge_attrs_2,
                # hyper_node_feats=hyper_node_feats,
                hyper_node_feats=torch.tensor([]).to(device),
                # svd_feats=[hyper_edge_feats_2, hyper_edge_feats_2_squre],
                # svd_attrs=[hyper_edge_attrs_2],
                # svd_feats=[hyper_edge_feats, hyper_edge_feats_1, hyper_edge_feats_2, hyper_edge_feats_squre, hyper_edge_feats_1_squre, hyper_edge_feats_2_squre],
                # svd_attrs=[hyper_edge_attrs, hyper_edge_attrs_1, hyper_edge_attrs_2],
                training=training,
            )
            
            # node_feats = node_feats + svd_embedding.unsqueeze(-1)
            # from IPython import embed
            # embed()
            # node_feats: [270, 512]、[270, 128]
            # node_feats before: [135, 256, 16]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            # node_feats after: [135, 4096] [135, 256]
            # from IPython import embed
            # embed()
            # exit(0)
            # try:
            #     node_feats = self.linear_6(torch.cat([node_feats, self.linear_3(svd_feats)], dim=1))
            # except:
            #     node_feats = self.linear_7(torch.cat([node_feats, self.linear_3(svd_feats)], dim=1))
                
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # for interaction2, product2, readout in zip(
        #     self.interactions, self.products, self.readouts
        # ):
        #     hyper_node_feats, sc = interaction2(
        #         node_attrs=data["node_attrs"],
        #         node_feats=hyper_node_feats,
        #         edge_attrs=edge_attrs,
        #         edge_feats=edge_feats,
        #         edge_index=data["edge_index"],
        #         svd_feats=svd_embedding,
        #         training=training,
        #     )
            
        #     # node_feats = node_feats + svd_embedding.unsqueeze(-1)
        #     # from IPython import embed
        #     # embed()
        #     # exit(0)
        #     # node_feats: [270, 512]、[270, 128]
        #     hyper_node_feats = product2(
        #         node_feats=hyper_node_feats, sc=sc, node_attrs=data["node_attrs"]
        #     )
        #     # from IPython import embed
        #     # embed()
        #     # exit(0)
        #     # node_feats = torch.cat([node_feats, svd_feat[0], svd_feat[1], svd_feat[2]], dim=1)
        #     node_feats_list.append(hyper_node_feats)
        #     node_es_list.append(readout(hyper_node_feats).squeeze(-1))  # {[n_nodes, ], }
        
        
        # Concatenate node features 
        # node_feats_out: [270, 640]
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # if training:
        #     node_feats_out = self.linear_2(torch.cat([node_feats_out, svd_embedding], dim=1))
        # node_feats_out = node_feats_out + svd_embedding
        
        # node_feats_out = self.linear_2(torch.cat([node_feats_out, self.linear_3(svd_feats)], dim=1))
        
        
        # Sum over interactions 
        # node_inter_es: [270]
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        # node_inter_es: [270]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph 
        # inter_e: [10]
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        # total_energy: [10]
        # node_energy: [270]
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        # from IPython import embed
        # embed()
        # forces: [270, 3]
        # virials、stress: None
        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        # from IPython import embed
        # embed()
        # exit(0)
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
        # print(111)
        # from IPython import embed
        # embed()
        # exit(0)
        return output


class BOTNet(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": compute_forces(
                energy=total_energy, positions=data.positions, training=training
            ),
        }

        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output
