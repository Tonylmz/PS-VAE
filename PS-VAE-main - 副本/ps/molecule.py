#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
from typing import Union

import networkx as nx
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol as RDKitMol
import numpy as np

from utils.chem_utils import smi2mol, mol2smi
from utils.chem_utils import get_submol, get_submol_atom_map


class SubgraphNode:
    '''
    The node representing a subgraph
    '''
    def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
        self.smiles = smiles
        self.pos = pos
        self.mol = smi2mol(smiles, kekulize, sanitize=False)
        # map atom idx in the molecule to atom idx in the subgraph (submol)
        self.atom_mapping = copy(atom_mapping)
    
    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos},
                    atom map: {self.atom_mapping}
                '''


class SubgraphEdge:
    '''
    Edges between two subgraphs
    '''
    def __init__(self, src: int, dst: int, edges: list):
        self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
        self.src = src
        self.dst = dst
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True
    
    def get_edges(self):
        return copy(self.edges)
    
    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src subgraph: {self.src}, dst subgraph: {self.dst},
                    atom bonds: {self.edges}
                '''


class Molecule(nx.Graph):
    '''molecule represented in subgraph-level'''

    def __init__(self, mol: Union[str, RDKitMol]=None, groups: list=None, kekulize: bool=False):
        super().__init__()
        if mol is None:
            return

        if isinstance(mol, str):
            smiles, rdkit_mol = mol, smi2mol(mol, kekulize)
        else:
            smiles, rdkit_mol = mol2smi(mol), mol
        self.graph['smiles'] = smiles
        # processing atoms
        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            subgraph_mol = get_submol(rdkit_mol, group, kekulize)
            subgraph_smi = mol2smi(subgraph_mol)
            atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, group, kekulize)
            node = SubgraphNode(subgraph_smi, pos, atom_mapping, kekulize)
            self.add_node(node)
        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            begin_subgraph_pos = aid2pos[begin]
            end_subgraph_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_subgraph_pos]['subgraph'].atom_mapping[begin]
            end_mapped = self.nodes[end_subgraph_pos]['subgraph'].atom_mapping[end]

            bond_type = bond.GetBondType()
            edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type))

        # add egdes into the graph
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = SubgraphEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> SubgraphEdge:
        # print(self[i][j])
        return self[i][j]['connects']
    
    def get_node(self, i) -> SubgraphNode:
        return self.nodes[i]['subgraph']

    def add_edge(self, edge: SubgraphEdge) -> None:
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: SubgraphNode) -> None:
        n = node.pos
        super().add_node(n, subgraph=node)

    def subgraph(self, nodes: list):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self):
        mol = Chem.RWMol()
        aid_mapping, order = {}, []
        # order = []
        # print(self.nodes)
        # add all the subgraphs to rwmol
        for n in self.nodes:
            # aid_mapping = {}
            subgraph = self.get_node(n)
            submol = subgraph.get_mol()
            # print(subgraph.smiles)
            # print("atom_mapping", subgraph.atom_mapping)
            local2global = {}
            for global_aid in subgraph.atom_mapping:
                local_aid = subgraph.atom_mapping[global_aid]
                local2global[local_aid] = global_aid
            # print("local2global", local2global)
            # print("order", order)
            for atom in submol.GetAtoms():
                if local2global[atom.GetIdx()] in order:
                    aid_mapping[(n, atom.GetIdx())] = order.index(local2global[atom.GetIdx()])
                    continue
                # print(atom.GetIdx(), atom.GetSymbol())
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(order)
                # aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
                order.append(local2global[atom.GetIdx()])
                # if local2global[atom.GetIdx()] not in order:
                #     order.append(local2global[atom.GetIdx()])
            # print(mol2smi(mol))
            # print("aid_mapping", aid_mapping)
            # print("order", order)
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                if mol.GetBondBetweenAtoms(begin, end) is None:
                    mol.AddBond(begin, end, bond.GetBondType())
                # print(begin, end, bond.GetBondType())
            # print(mol2smi(mol))
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol)
        # for bond in mol.GetBonds():
        #     begin, end, bond_type = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
        #     print(begin, end, bond_type)
        # for src, dst in self.edges:
        #     subgraph_edge = self.get_edge(src, dst)
        #     pid_src, pid_dst = subgraph_edge.src, subgraph_edge.dst
        #     for begin, end, bond_type in subgraph_edge.edges:
        #         begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
        #         mol.AddBond(begin, end, bond_type)
        # print(mol.GetNumAtoms())
        # print(mol.GetNumBonds())
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol)
        # for bond in mol.GetBonds():
        #     begin, end, bond_type = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
        #     print(begin, end, bond_type)
        mol = mol.GetMol()
        new_order = [-1 for _ in order]
        for cur_i, ordered_i in enumerate(order):
            new_order[ordered_i] = cur_i
        # print(new_order)
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol)
        # for bond in mol.GetBonds():
        #     begin, end, bond_type = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
        #     print(begin, end, bond_type)
        mol = Chem.RenumberAtoms(mol, new_order)
        # sanitize, we need to handle mal-formed N+
        mol.UpdatePropertyCache(strict=False)
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol)
        # for bond in mol.GetBonds():
        #     begin, end, bond_type = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
        #     print(begin, end, bond_type)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:  # no problem
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':  # for N+, we need to set its formal charge
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol)
        # for bond in mol.GetBonds():
        #     begin, end, bond_type = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
        #     print(begin, end, bond_type)
        return mol

    def to_SVG(self, path: str, size: tuple=(200, 200), add_idx=False) -> str:
        # save the subgraph-level molecule to an SVG image
        # return the content of svg in string format
        mol = self.to_rdkit_mol()
        if add_idx:  # this will produce an ugly figure
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                atom.SetAtomMapNum(i)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        view = rdMolDraw2D.MolDraw2DSVG(*size)
        option = view.drawOptions()
        option.legendFontSize = 18
        option.bondLineWidth = 1
        option.highlightBondWidthMultiplier = 20
        sg_atoms, sg_bonds = [], []
        # atom2subgraph, atom_color, bond_color = {}, {}, {}
        atom_color, bond_color, atom_count, bond_count = {}, {}, {}, {}
        # atoms in each subgraph
        # print("self.nodes", self.nodes)
        for i in self.nodes:
            # sg_atoms, sg_bonds = [], []
            # atom2subgraph, atom_color, bond_color = {}, {}, {}
            node = self.get_node(i)
            # random color in rgb. mix with white to obtain soft colors
            color = tuple(((np.random.rand(3) + 1)/ 2).tolist())
            # print("atom_mapping", node.atom_mapping)
            # submol = get_submol(mol, list(node.atom_mapping.keys()), kekulize=False)
            # print(mol2smi(submol))
            for atom_id in node.atom_mapping:
                if atom_id not in atom_count:
                    atom_count[atom_id] = 1
                    sg_atoms.append(atom_id)
                    atom_color[atom_id] = color
                else:
                    atom_color[atom_id] = tuple((np.array(atom_color[atom_id]) * atom_count[atom_id] + np.array(color)) / (atom_count[atom_id] + 1))
                    atom_count[atom_id] += 1
                    
            # for atom_id in node.atom_mapping:
            #     if atom_id in atom2subgraph:
            #         continue
            #     sg_atoms.append(atom_id)
            #     atom2subgraph[atom_id] = i
            #     atom_color[atom_id] = color
            for bond_id in range(mol.GetNumBonds()):
                bond = mol.GetBondWithIdx(bond_id)
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if begin in node.atom_mapping and end in node.atom_mapping:
                    if bond_id not in bond_count:
                        bond_count[bond_id] = 1
                        sg_bonds.append(bond_id)
                        bond_color[bond_id] = color
                    else:
                        bond_color[bond_id] = tuple((np.array(bond_color[bond_id]) * bond_count[bond_id] + np.array(color)) / (bond_count[bond_id] + 1))
                        bond_count[bond_id] += 1
            # print("sg_atoms", sg_atoms)
        # bonds in each subgraph
        # print("atom2subgraph", atom2subgraph)
        # for bond_id in range(mol.GetNumBonds()):
        #     bond = mol.GetBondWithIdx(bond_id)
        #     begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #     # print("begin, end, bond_type", begin, end, bond.GetBondType())
        #     if atom2subgraph[begin] == atom2subgraph[end]:
        #         sg_bonds.append(bond_id)
        #         bond_color[bond_id] = atom_color[begin]
        view.DrawMolecules([tm], highlightAtoms=[sg_atoms], \
                        highlightBonds=[sg_bonds], highlightAtomColors=[atom_color], \
                        highlightBondColors=[bond_color])
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(path, 'w') as fout:
            fout.write(svg)
        # for atom in mol.GetAtoms():
        #     idx, symbol = atom.GetIdx(), atom.GetSymbol()
        #     print(idx, symbol, atom_color[idx])
        return svg

    def to_smiles(self):
        rdkit_mol = self.to_rdkit_mol()
        return mol2smi(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            # print(self.get_node(node).get_atom_mapping())
            desc += f'{ni}:{self.get_node(node)}\n'
        if len(self.nodes) > 1:
            desc += 'intersect_nodes: \n'
            intersect_nodes = []
            for ni1, node1 in enumerate(self.nodes):
                for ni2, node2 in enumerate(self.nodes):
                    if ni1 != ni2 and {ni1, ni2} not in intersect_nodes:
                        intersect_nodes.append({ni1, ni2})
                        key_src = self.get_node(node1).get_atom_mapping()
                        key_dst = self.get_node(node2).get_atom_mapping()
                        intersection = set(key_src.keys()).intersection(set(key_dst.keys()))
                        if len(intersection) > 0:
                            aids, symbols = [], []
                            for aid in intersection:
                                symbol = self.get_node(node1).get_mol().GetAtomWithIdx(key_src[aid]).GetSymbol()
                                aids.append(aid)
                                symbols.append(symbol)
                                # desc += f'{src}-{dst}:{aid, symbol}\n'
                            # desc += f'{node1}-{node2}:{"atom id:", aids, "atom symbol:", symbols}\n'
                            desc += f'''
                            {node1}-{node2}:{"atom id:", aids, "atom symbol:", symbols}\n
                            '''
        # desc += 'intersect_nodes: \n'
        # for src, dst in self.edges:
        #     key_src = self.get_node(src).get_atom_mapping()
        #     key_dst = self.get_node(dst).get_atom_mapping()
        #     intersection = set(key_src.keys()).intersection(set(key_dst.keys()))
        #     aids, symbols = [], []
        #     for aid in intersection:
        #         symbol = self.get_node(src).get_mol().GetAtomWithIdx(key_src[aid]).GetSymbol()
        #         aids.append(aid)
        #         symbols.append(symbol)
        #         # desc += f'{src}-{dst}:{aid, symbol}\n'
        #         # desc += f'{src}-{dst}:' + f'''
        #         # {"atom id:", aid, "atom symbol:", symbol}\n
        #         # '''
        #     desc += f'''
        #     {src}-{dst}:{"atom ids:", aids, "atom symbols:", symbols}\n
        #     '''
            # desc += f'{src}-{dst}:{intersection}\n'
            # desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        # desc += 'edges: \n'
        # for src, dst in self.edges:
        #     desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc
    
    def get_res(self):
        desc = {}
        desc['nodes'] = []
        for ni, node in enumerate(self.nodes):
            desc['nodes'].append({ni:{"atom map": self.get_node(node).get_atom_mapping(), "smiles": self.get_node(node).smiles, "position": self.get_node(node).pos}})
        if len(self.nodes) > 1:
            desc['intersect_nodes'] = []
            s = []
            for ni1, node1 in enumerate(self.nodes):
                for ni2, node2 in enumerate(self.nodes):
                    if ni1 != ni2 and {ni1, ni2} not in s:
                        s.append({ni1, ni2})
                        key_src = self.get_node(node1).get_atom_mapping()
                        key_dst = self.get_node(node2).get_atom_mapping()
                        intersection = set(key_src.keys()).intersection(set(key_dst.keys()))
                        if len(intersection) > 0:
                            aids, symbols = [], []
                            for aid in intersection:
                                symbol = self.get_node(node1).get_mol().GetAtomWithIdx(key_src[aid]).GetSymbol()
                                aids.append(aid)
                                symbols.append(symbol)
                            desc['intersect_nodes'].append({f'{node1}-{node2}':{"atom id": aids, "atom symbol": symbols}})
        return desc