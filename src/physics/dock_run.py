#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""Stage 2 of the Family G ligand-affinity library (`docking_library.py`): AutoDock Vina 1.2.7 in the
docking environment. Receptors to PDBQT with Open Babel (rigid; Vina's scoring has no partial-charge
term), ligands from SMILES with RDKit (ETKDG embedding, MMFF relaxation) and Meeko: DAPI (dication,
both amidines protonated) for the minor groove and proflavine (ring-N protonated cation) for the
intercalation site. Best pose's Vina score per site, exhaustiveness 32, fixed seed."""
import argparse, glob, json, os, subprocess, sys, tempfile

LIGANDS = {"minor": ("DAPI", "NC(=[NH2+])c1ccc(cc1)-c1cc2ccc(cc2[nH]1)C(N)=[NH2+]", 22.0),
           "intercalation": ("proflavine", "Nc1ccc2cc3ccc(N)cc3[nH+]c2c1", 18.0)}


def ligand_pdbqt(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7); AllChem.MMFFOptimizeMolecule(m)
    setups = MoleculePreparation().prepare(m)
    s, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(err)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receptors", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    from vina import Vina
    os.makedirs(a.out, exist_ok=True)
    ligs = {site: (name, ligand_pdbqt(smi), box) for site, (name, smi, box) in LIGANDS.items()}
    metas = sorted(glob.glob(os.path.join(a.receptors, "*.json")))[a.shard::a.n_shards]
    for meta in metas:
        ctx = os.path.basename(meta)[:-5]
        out = os.path.join(a.out, f"{ctx}.json")
        if os.path.exists(out):
            continue
        r = json.load(open(meta)); res = {"context": ctx}
        for site, v in r.items():
            name, lig, box = ligs[site]
            w = tempfile.mkdtemp(); rec = os.path.join(w, "rec.pdbqt")
            subprocess.run(["obabel", v["pdb"], "-xr", "-O", rec], capture_output=True, check=True)
            vn = Vina(sf_name="vina", seed=11, verbosity=0)
            vn.set_receptor(rec); vn.set_ligand_from_string(lig)
            vn.compute_vina_maps(center=v["centre"], box_size=[box, box, box])
            vn.dock(exhaustiveness=32, n_poses=9)
            e = vn.energies(n_poses=1)
            res[f"{site}_ligand"] = name; res[f"{site}_vina_kcal_mol"] = float(e[0][0])
        json.dump(res, open(out, "w"), indent=1)
        print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
