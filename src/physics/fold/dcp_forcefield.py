#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""OpenMM residue templates for N3-protonated deoxycytidine (DCP, with 5'/3'/both-terminal
variants DCP5/DCP3/DCPN), the third-strand C+ of pyrimidine-motif triplets and the C+ of i-motif
C.C+ pairs. Each is AMBER OL15's DC template of the same terminal variant with the base charges
of `physics/data/dcp_charges.json` (RESP, `src/qc/resp_cplus.py`) and an H3 on N3; N3 takes the
type of a protonated ring nitrogen (DNA-NA, as thymine's N3) and H3 that of thymine's H3 (DNA-H),
so every bonded term exists in OL15. `write_xml(path)` writes the templates; load the file after
amber14/DNA.OL15.xml.
"""
import json
import os
import xml.etree.ElementTree as ET

import openmm.app as app

HERE = os.path.dirname(os.path.abspath(__file__))
CHARGES = os.path.join(os.path.dirname(HERE), "data", "dcp_charges.json")
VARIANTS = {"DC": "DCP", "DC5": "DCP5", "DC3": "DCP3", "DCN": "DCPN"}


def write_xml(path: str) -> str:
    q = json.load(open(CHARGES))["base_charges"]
    src = os.path.join(os.path.dirname(app.__file__), "data", "amber14", "DNA.OL15.xml")
    root = ET.parse(src).getroot()
    out = ET.Element("ForceField"); res_el = ET.SubElement(out, "Residues")
    for r in root.find("Residues"):
        name = r.get("name")
        if name not in VARIANTS:
            continue
        new = ET.SubElement(res_el, "Residue", name=VARIANTS[name])
        for child in r:
            if child.tag == "Atom":
                a = dict(child.attrib)
                if a["name"] in q:
                    a["charge"] = f"{q[a['name']]:.4f}"
                if a["name"] == "N3":
                    a["type"] = "DNA-NA"
                ET.SubElement(new, "Atom", a)
            else:
                ET.SubElement(new, child.tag, dict(child.attrib))
        ET.SubElement(new, "Atom", {"charge": f"{q['H3']:.4f}", "name": "H3", "type": "DNA-H"})
        ET.SubElement(new, "Bond", {"atomName1": "N3", "atomName2": "H3"})
    ET.ElementTree(out).write(path)
    return path


if __name__ == "__main__":
    import sys
    p = write_xml(sys.argv[1] if len(sys.argv) > 1 else "dcp.xml")
    ff = app.ForceField("amber14/DNA.OL15.xml", p)
    print("loaded", p)
