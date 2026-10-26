#!/usr/bin/env python3
"""Prepare ChEMBL ADRA2B references and an independent PubChem candidate pool."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'reports/adra2b_screening_20261026'
SRC = OUT / 'sources'
CFG = PROJECT / 'configs/adra2b_screening_20261026.json'


@lru_cache(None)
def identity(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('Invalid SMILES')
    m = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(m))
    Chem.SanitizeMol(m)
    s = Chem.MolToSmiles(m, isomericSmiles=True)
    key = Chem.MolToInchiKey(m)
    return s, key, key.split('-')[0], Chem.AddHs(m).GetNumAtoms()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RDLogger.DisableLog('rdApp.warning')
    cfg = json.loads(CFG.read_text())
    uni = json.loads((SRC / 'uniprot_P18089.json').read_text())
    assert uni['primaryAccession'] == cfg['target']
    sequence = uni['sequence']['value']
    assay = pd.read_csv(SRC / 'pubchem_1961694_data.csv')
    assay = assay.loc[pd.to_numeric(assay.PUBCHEM_RESULT_TAG, errors='coerce').notna()].copy()
    assert len(assay) >= 1000
    candidates = []
    cutoff = cfg['external_activity_cutoff_nM']
    for raw in assay.to_dict('records'):
        reasons = []
        s = full = key = ''
        try:
            s, full, key, atoms = identity(raw['PUBCHEM_EXT_DATASOURCE_SMILES'])
            if atoms > 512:
                reasons.append('encoder_size_limit')
        except (TypeError, ValueError, RuntimeError):
            reasons.append('structure')
        try:
            value = float(raw['Standard Value'])
        except (ValueError, TypeError):
            value = np.nan
        relation = raw['Standard Relation']
        outcome = 'Uncertain'
        if raw['Standard Type'] != 'AC50' or raw['Standard Units'] != 'nM' or not np.isfinite(value) or value <= 0:
            reasons.append('endpoint')
        elif relation == '=':
            outcome = 'Active' if value <= cutoff else 'Inactive'
        elif relation == '>' and value >= cutoff:
            outcome = 'Inactive'
        elif relation in ('<', '<=') and value <= cutoff:
            outcome = 'Active'
        elif relation == '>=' and value > cutoff:
            outcome = 'Inactive'
        else:
            reasons.append('threshold_ambiguous')
        if s in cfg.get('failed_standard_conformer_smiles', []):
            reasons.append('standard_conformer_failure')
        tag = int(raw['PUBCHEM_RESULT_TAG'])
        sid = int(raw['PUBCHEM_SID'])
        candidates.append(dict(source_id=f'aid1961694_sid{sid}_row{tag}',
                               cid=raw['PUBCHEM_CID'], sid=sid, raw_result_tag=tag,
                               Canonical_SMILES=s, parent_inchikey=full, parent_key=key,
                               outcome=outcome, activity_score=raw['PUBCHEM_ACTIVITY_SCORE'],
                               deposited_outcome=raw['PUBCHEM_ACTIVITY_OUTCOME'],
                               AC50_nM=value, relation=relation,
                               exclusion_reason='|'.join(reasons)))
    candidates = pd.DataFrame(candidates).sort_values('source_id').reset_index(drop=True)
    repeated = candidates.parent_key.ne('') & candidates.duplicated('parent_key', keep=False)
    candidates.loc[repeated, 'exclusion_reason'] = candidates.loc[repeated, 'exclusion_reason'].apply(
        lambda x: '|'.join(filter(None, [x, 'repeated_parent'])))
    candidates['included'] = candidates.exclusion_reason.eq('')
    candidates.to_csv(OUT / 'candidate_audit.csv', index=False)
    query = candidates.loc[candidates.included].copy().reset_index(drop=True)
    assert len(query) >= 100 and query.outcome.value_counts().min() >= 10
    query['row_id'] = np.arange(len(query))
    query['drug_id'] = query.parent_key
    query['protein_id'] = cfg['target']
    query['Sequence'] = sequence
    holdout_keys = set(candidates.parent_key)
    source_files = sorted(SRC.glob('chembl_adra2b_ki_page*.json'))
    activities = [a for f in source_files for a in json.loads(f.read_text())['activities']]
    assays = {a['assay_chembl_id']: a for f in sorted(SRC.glob('chembl_adra2b_assays_*.json'))
              for a in json.loads(f.read_text())['assays']}
    records = []
    for act in activities:
        info = assays[act['assay_chembl_id']]
        reasons = []
        if act['target_chembl_id'] != 'CHEMBL1942' or str(act['target_tax_id']) != '9606':
            reasons.append('target')
        if act['standard_type'] != 'Ki' or act['standard_relation'] != '=' or act['standard_units'] != 'nM':
            reasons.append('endpoint')
        try:
            value = float(act['standard_value'])
            if not np.isfinite(value) or value <= 0:
                reasons.append('value')
        except (TypeError, ValueError):
            value = np.nan
            reasons.append('value')
        if info['confidence_score'] != 9 or info['assay_type'] != 'B':
            reasons.append('assay_assignment')
        if act['document_chembl_id'] == cfg['external_document']:
            reasons.append('external_study')
        if info['src_id'] != 1:
            reasons.append('non_literature_source')
        if info['variant_sequence'] is not None or act['assay_variant_mutation']:
            reasons.append('annotated_variant')
        if act['data_validity_comment'] not in (None, 'Manually validated') or act['potential_duplicate']:
            reasons.append('data_quality')
        s = full = key = ''
        try:
            s, full, key, atoms = identity(act['canonical_smiles'])
            if atoms > 512:
                reasons.append('encoder_size_limit')
        except (TypeError, ValueError, RuntimeError):
            reasons.append('structure')
        if key in holdout_keys:
            reasons.append('external_candidate')
        if s in cfg.get('failed_standard_conformer_smiles', []):
            reasons.append('standard_conformer_failure')
        records.append(dict(activity_id=act['activity_id'], assay_id=act['assay_chembl_id'],
                            document_id=act['document_chembl_id'], year=act['document_year'],
                            molecule_id=act['molecule_chembl_id'], source_id=act['src_id'],
                            Canonical_SMILES=s, parent_inchikey=full, parent_key=key,
                            Ki_nM=value, pKi=9 - np.log10(value) if value > 0 else np.nan,
                            exclusion_reason='|'.join(reasons)))
    audit = pd.DataFrame(records)
    audit.to_csv(OUT / 'training_record_audit.csv', index=False)
    eligible = audit.loc[audit.exclusion_reason.eq('')]
    training = eligible.groupby(['Canonical_SMILES', 'parent_inchikey', 'parent_key'], sort=True).agg(
        pKi=('pKi', 'median'), measurements=('activity_id', 'size'),
        activity_ids=('activity_id', lambda x: '|'.join(str(i) for i in x)),
        document_ids=('document_id', lambda x: '|'.join(sorted(set(x)))),
        assay_ids=('assay_id', lambda x: '|'.join(sorted(set(x)))))
    training = training.reset_index()
    training['row_id'] = np.arange(len(training))
    training['drug_id'] = training.parent_key
    training['protein_id'] = cfg['target']
    training['Sequence'] = sequence
    assert len(training) >= 200 and training.pKi.std() > .5
    assert not set(training.parent_key) & holdout_keys
    groups = np.array(sorted(training.parent_key.unique()))
    np.random.RandomState(cfg['partition_seed']).shuffle(groups)
    chunks = np.array_split(groups, cfg['validation_folds'])
    splits = []
    for fold, val_groups in enumerate(chunks):
        frame = training[['row_id', 'parent_key']].copy()
        frame['fold'] = fold
        frame['split'] = np.where(frame.parent_key.isin(val_groups), 'val', 'train')
        assert not set(frame.loc[frame.split.eq('train'), 'parent_key']) & set(val_groups)
        splits.append(frame)
    training.to_csv(OUT / 'training_table.csv', index=False)
    pd.concat(splits, ignore_index=True).to_csv(OUT / 'partitions.csv', index=False)
    qcols = ['row_id', 'source_id', 'drug_id', 'protein_id', 'Canonical_SMILES', 'Sequence', 'parent_key']
    query[qcols].to_csv(OUT / 'query_table.csv', index=False)
    query[['row_id', 'source_id', 'cid', 'sid', 'raw_result_tag', 'outcome', 'activity_score', 'parent_key', 'deposited_outcome', 'AC50_nM', 'relation']].to_csv(OUT / 'labels.csv', index=False)
    pd.concat([training[qcols[:1] + ['drug_id', 'protein_id', 'Canonical_SMILES', 'Sequence']],
               query[['row_id', 'drug_id', 'protein_id', 'Canonical_SMILES', 'Sequence']]], ignore_index=True).to_csv(
                   OUT / 'feature_table.csv', index=False)
    report = dict(target=cfg['target'], target_family=cfg['family'], source_activity_records=len(audit),
                  eligible_measurements=len(eligible), training_compounds=len(training),
                  training_parent_groups=training.parent_key.nunique(),
                  documents=eligible.document_id.nunique(), assays=eligible.assay_id.nunique(),
                  pKi_range=[float(training.pKi.min()), float(training.pKi.max())],
                  external_source_records=len(candidates), external_source_compounds=candidates.sid.nunique(), external_compounds=len(query),
                  external_outcomes=query.outcome.value_counts().to_dict(),
                  external_exclusion_reasons=candidates.exclusion_reason.value_counts().to_dict(),
                  training_measurements_removed_for_external_overlap=int(audit.exclusion_reason.str.contains('external_candidate').sum()),
                  conformer_failure_compounds=audit.loc[audit.exclusion_reason.str.contains('standard_conformer_failure'), 'Canonical_SMILES'].nunique(),
                  training_external_parent_overlap=0, config_sha256=digest(CFG))
    (OUT / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    files = [CFG, Path(__file__), SRC / 'provenance.json'] + [p for p in SRC.glob('*.json')] + [SRC / 'pubchem_1961694_data.csv']
    (OUT / 'preparation_inputs.json').write_text(json.dumps(
        {str(p.relative_to(PROJECT)): digest(p) for p in sorted(set(files))}, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
