from mol_bpe import Tokenizer
from rdkit import Chem
from rdkit.Chem import Draw

smiles = 'C[C@@]1(C)O[C@]2(C)[C@H](O)[C@H]12'
# smiles = 'C#C[C@@H]1[C@@H]2C[C@@H]2[C@@]1(C)O'
# smiles = 'C[CH+]1N=C([NH-])ON=C1F'
# construct a tokenizer from a given vocabulary
tokenizer = Tokenizer(r'E:\Study\glad\projects\PS-VAE-main\PS-VAE-main\ps\output\vocab.txt')
# piece-level decomposition
mol = tokenizer(smiles)
print('subgraph level decomposition:')
print(mol)
mol.to_SVG('example2.svg')
# peptide_mol = Chem.MolFromSmiles(smiles)
# peptide_mol