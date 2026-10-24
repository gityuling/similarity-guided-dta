#!/usr/bin/env python3
"""Source/identity audit for two downstream tasks, without reading predictions."""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from bib_common import PROJECT, record, write_once

OUT = PROJECT / 'reports/bib_downstream_20261022'
SRC = OUT / 'sources'


@lru_cache(None)
def structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid structure')
    mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    Chem.SanitizeMol(mol)
    full = Chem.MolToInchiKey(mol)
    return (Chem.MolToSmiles(mol, isomericSmiles=True), full, full.split('-')[0],
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))


def main():
    RDLogger.DisableLog('rdApp.warning')
    inputs = [SRC/'ligunity_fep_labels.json', SRC/'pubchem_1963317_data.csv',
              SRC/'pubchem_1963317_description.json', SRC/'ligunity_git_tree.json',
              SRC/'ligunity_article.xml', SRC/'abl1_article.xml', __file__]
    parents, scaffolds = {}, {}
    for dataset in ('kiba', 'davis'):
        path = PROJECT/f'data/processed/{dataset}_dtiam_protocol/table.csv'
        inputs.append(path)
        table = pd.read_csv(path)
        values = [structure(s) for s in table.Canonical_SMILES.unique()]
        parents[dataset] = {v[2] for v in values}
        scaffolds[dataset] = {v[3] for v in values if v[3]}
        if dataset == 'kiba':
            proteins = table[['protein_id', 'Sequence']].drop_duplicates().set_index('protein_id')
    previous = [
        ('bib_selectivity_20261021/compound_audit.csv', 'parent_key', True),
        ('bib_g2cp_dta_20261019/annotation_review.csv', 'parent_key', True),
        ('bib_nanobret_20261018/compound_audit.csv', 'Canonical_SMILES', False),
        ('bib_cellular_case_external_20261018/query_table.csv', 'Canonical_SMILES', False),
        ('bib_cellular_case_reviewed_20261018/candidate_scores_before_endpoints.csv', 'chemical_identity', False),
    ]
    for path, col, is_key in previous:
        p = PROJECT/'reports'/path; inputs.append(p)
        # Read structure column only; never access historical model or endpoint scores.
        values = pd.read_csv(p, usecols=[col])[col].dropna().unique()
        parents[path.split('/')[0]] = set(values) if is_key else {structure(s)[2] for s in values}

    def audit(row, smiles):
        canonical, full, parent, scaffold = structure(smiles)
        row.update(Canonical_SMILES=canonical, full_parent_inchikey=full, parent_key=parent,
                   scaffold=scaffold, drug_id=parent)
        seen = []
        for name, keys in parents.items():
            row['seen_'+name] = parent in keys
            if parent in keys: seen.append(name)
        row['overlap_sources'] = '|'.join(seen)
        for name, keys in scaffolds.items(): row['scaffold_seen_'+name] = bool(scaffold) and scaffold in keys
        row['exclusion_reason'] = 'prior_compound_exposure' if seen else ''
        return row

    records, source_checks, inventory = [], [], []
    for group in json.loads((SRC/'ligunity_fep_labels.json').read_text()):
        series, protein = group['pockets'][0], group['uniprot']
        supported = protein in proteins.index and proteins.loc[protein, 'Sequence'] == group['sequence']
        inventory.append(dict(series=series, protein_id=protein, source_rows=len(group['ligands']), exact_kiba_sequence=supported))
        sdf_map = {}
        if supported:
            files = list(SRC.glob(series+'_*ligands.sdf'))
            assert len(files) == 1
            inputs += files
            for ordinal, mol in enumerate(Chem.SDMolSupplier(str(files[0])), 1):
                assert mol is not None
                if not mol.HasProp('r_exp_dg'): continue
                identity = structure(Chem.MolToSmiles(mol))[1]
                sdf_map.setdefault(identity, []).append((ordinal, float(mol.GetProp('r_exp_dg'))))
        for ordinal, ligand in enumerate(group['ligands'], 1):
            row = audit(dict(task='B', cohort=series, source_ordinal=ordinal,
                             source_id=f'{series}_{ordinal:03d}', protein_id=protein,
                             Sequence=group['sequence'], endpoint_score=float(ligand['act']),
                             source_smiles=ligand['smi'], outcome='quantitative'), ligand['smi'])
            if not supported:
                row['exclusion_reason'] = 'target_not_exactly_supported'
            else:
                matches = sdf_map[row['full_parent_inchikey']]
                dg = [v for _, v in matches]
                assert max(dg)-min(dg) < 0.001, 'Conflicting SDF measurements'
                error = max(abs(v+1.364*ligand['act']) for v in dg)
                assert error < 0.0001, 'Experimental label transformation mismatch'
                source_checks.append(dict(source_id=row['source_id'], sdf=files[0].name,
                    sdf_ordinals='|'.join(str(i) for i,_ in matches), experimental_dg=dg[0],
                    json_score=ligand['act'], max_abs_dg_error=error,
                    identity_match='full standardized parent InChIKey including stereochemistry'))
            records.append(row)

    assay = pd.read_csv(SRC/'pubchem_1963317_data.csv', low_memory=False)
    assay = assay.loc[pd.to_numeric(assay.PUBCHEM_RESULT_TAG, errors='coerce').notna()]
    assert len(assay) == 128 and 'P00519' in proteins.index
    for _, r in assay.iterrows():
        outcomes = [int(r[f'Rep{i}ABLNHIBITKGP-Outcome']) for i in range(1,5)]
        active_replicates = sum(v == 2 for v in outcomes)
        noninactive_replicates = sum(v in (2,3) for v in outcomes)
        score = float(r.PUBCHEM_ACTIVITY_SCORE)
        # Deposited aggregate counts both active (2) and inconclusive (3) repeats.
        # Preserve that outcome, and separately retain a stricter active-repeat sensitivity.
        assert score == (100 if noninactive_replicates >= 3 else 30 if noninactive_replicates else 0)
        outcome = r.PUBCHEM_ACTIVITY_OUTCOME
        assert outcome == {100:'Active',30:'Inconclusive',0:'Inactive'}[int(score)]
        records.append(audit(dict(task='A', cohort='ABL1_AID1963317',
            source_ordinal=int(r.PUBCHEM_RESULT_TAG), source_id='aid1963317_sid'+str(int(r.PUBCHEM_SID)),
            protein_id='P00519', Sequence=proteins.loc['P00519','Sequence'],
            source_smiles=r.PUBCHEM_EXT_DATASOURCE_SMILES, outcome=outcome,
            active_replicates=active_replicates, noninactive_replicates=noninactive_replicates,
            strict_outcome='Active' if active_replicates>=3 else ('Inactive' if noninactive_replicates==0 else 'Inconclusive'),
            endpoint_score=np.nan), r.PUBCHEM_EXT_DATASOURCE_SMILES))

    data = pd.DataFrame(records)
    duplicate = data.duplicated(['task','cohort','parent_key'], keep=False)
    data.loc[duplicate & data.exclusion_reason.eq(''), 'exclusion_reason'] = 'duplicate_parent_within_cohort'
    # Do not split a shared compound between the two new tasks.
    cross = data.groupby('parent_key').task.nunique()
    shared = set(cross[cross > 1].index)
    data.loc[data.parent_key.isin(shared) & data.exclusion_reason.eq(''), 'exclusion_reason'] = 'shared_between_new_tasks'
    coverage = []
    for (task, cohort), g in data.groupby(['task','cohort'], sort=True):
        eligible = g.loc[g.exclusion_reason.eq('')]
        if task == 'B':
            dynamic_range = float(eligible.endpoint_score.max()-eligible.endpoint_score.min()) if len(eligible) else 0
            feasible = len(eligible) >= 10 and dynamic_range >= 1.0
        else:
            dynamic_range = None
            feasible = len(eligible) >= 30 and (eligible.outcome=='Active').sum() >= 5 and (eligible.outcome=='Inactive').sum() >= 5
        coverage.append(dict(task=task, cohort=cohort, source_n=len(g), eligible_before_size_gate=len(eligible),
            active_n=int((eligible.outcome=='Active').sum()), inactive_n=int((eligible.outcome=='Inactive').sum()),
            inconclusive_n=int((eligible.outcome=='Inconclusive').sum()), score_range=dynamic_range,
            included=bool(feasible), scaffold_kiba_seen=int(eligible.scaffold_seen_kiba.sum())))
        if not feasible:
            data.loc[eligible.index, 'exclusion_reason'] = 'insufficient_cohort_size_or_dynamic_range'
    data['included'] = data.exclusion_reason.eq('')
    selected = data.loc[data.included].sort_values(['task','cohort','source_id']).copy()
    selected.insert(0, 'row_id', np.arange(len(selected)))
    query_cols = ['row_id','task','cohort','source_id','drug_id','Canonical_SMILES','protein_id','Sequence']
    labels = selected[['row_id','task','cohort','source_id','endpoint_score','outcome','strict_outcome','parent_key','scaffold_seen_kiba','scaffold_seen_davis']]
    for name, frame in [('source_compound_audit.csv',data),('cohort_coverage.csv',pd.DataFrame(coverage)),
                        ('series_inventory.csv',pd.DataFrame(inventory)),('experimental_label_verification.csv',pd.DataFrame(source_checks)),
                        ('query_table.csv',selected[query_cols]),('labels.csv',labels)]:
        write_once(OUT/name, frame.to_csv(index=False))
    from pathlib import Path
    provenance = [record(Path(p)) for p in inputs]
    write_once(OUT/'preparation_inputs.json', json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(coverage).to_string(index=False))
    print('Selected:',len(selected),'unique compounds',selected.drug_id.nunique())


if __name__ == '__main__': main()
