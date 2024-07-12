from mol_bpe import Tokenizer
from rdkit import Chem
from rdkit.Chem import Draw

# smiles = 'N#CC#N'
# smiles = 'C[C@@]1(C)O[C@]2(C)[C@H](O)[C@H]12'
# smiles = 'C#C[C@@H]1[C@@H]2C[C@@H]2[C@@]1(C)O'
# smiles = 'C[CH+]1N=C([NH-])ON=C1F'
# construct a tokenizer from a given vocabulary
# smiles = 'CC(=O)CC(=O)C'
smiles = 'C1C([H])=NC(=C(C=1[H])OC([H])([H])C1C([H])=C([H])C([H])=C([H])C=1[H])N([H])[H]'
tokenizer = Tokenizer(r'E:\Study\glad\projects\PS-VAE-main\PS-VAE-main - 副本\ps\output\vocab_new2.txt')
# piece-level decomposition
mol = tokenizer(smiles)
print('subgraph level decomposition:')
# mol.get_res()
print(mol.get_res())
mol.to_SVG('example3.svg')
# peptide_mol = Chem.MolFromSmiles(smiles)
# peptide_mol