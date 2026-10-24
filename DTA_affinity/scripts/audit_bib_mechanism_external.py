#!/usr/bin/env python3
"""Structure/metadata/assay coverage only; never load external model scores."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from bib_g2cp_common import DATA, OUT, PROJECT, ROOT, record, save_json, write_once
from prepare_bib_nanobret import parent_key

DEST=PROJECT/'reports/bib_mechanism_20261021'
SRC=DEST/'sources'
CELLS=['A375','A549','MCF7','PC3','HT29']


@lru_cache(maxsize=None)
def structure(smiles):
    try:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None: return '', '', ''
        canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
        return canonical,parent_key(canonical),MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
    except (ValueError,RuntimeError): return '', '', ''


def main():
    RDLogger.DisableLog('rdApp.*')
    paths=[SRC/'Tool_compounds.csv',SRC/'reinecke_matrix.csv',SRC/'reinecke_target_annotation.csv',
           OUT/'chemical_split.csv',OUT/'candidates.csv',OUT/'annotation_review.csv',
           PROJECT/'reports/bib_nanobret_20261018/compound_audit.csv',
           PROJECT/'reports/bib_nanobret_20261018/protein_annotation_audit.csv',
           PROJECT/'reports/bib_cellular_case_reviewed_20261018/candidate_scores_before_endpoints.csv',
           ROOT/'UniPert-G2CP_reproduce/G2CP/data/processed_CMAP_compound_info.csv',Path(__file__)]
    sources=[record(p) for p in paths]
    compounds=pd.read_csv(paths[0],keep_default_na=False)
    matrix=pd.read_csv(paths[1],keep_default_na=False).set_index('Gene Name')
    annotations=pd.read_csv(paths[2],keep_default_na=False).set_index('Gene')
    splits=pd.read_csv(paths[3],keep_default_na=False)
    prior_candidates=pd.read_csv(paths[4],keep_default_na=False)
    preliminary=pd.read_csv(paths[5],keep_default_na=False)
    old_braf=pd.read_csv(paths[8],usecols=['chemical_identity'])
    braf={structure(s)[1] for s in old_braf.chemical_identity}
    nb=pd.read_csv(paths[6]); nb_smiles=next(c for c in nb.columns if c=='Canonical_SMILES')
    nano={structure(s)[1] for s in nb[nb_smiles]}
    old59=set(prior_candidates.parent_key);old68=set(preliminary.parent_key)
    cache=pd.read_csv(paths[9],keep_default_na=False)
    lincs=defaultdict(list)
    for r in cache.itertuples():
        exact,parent,_=structure(r.canonical_smiles)
        if parent: lincs[parent].append((r.pert_id,exact))
    split_lookup=splits.groupby('parent_key').split.agg(lambda v:'|'.join(sorted(set(v)))).to_dict()
    training={};scaffolds={}
    for dataset in ['kiba','davis']:
        p=PROJECT/f'data/processed/{dataset}_dtiam_protocol/table.csv';sources.append(record(p))
        values=[structure(s) for s in pd.read_csv(p,usecols=['Canonical_SMILES']).Canonical_SMILES.unique()]
        training[dataset]={v[1] for v in values};scaffolds[dataset]={v[2] for v in values}
    protein=pd.read_csv(paths[7],keep_default_na=False)
    gene_to_protein=defaultdict(set)
    for row in protein.itertuples(index=False,name=None):
        v=dict(zip(protein.columns,row))
        if str(v['canonical_sequence_matches']).lower()!='true':continue
        for gene in v['Gene Names'].split():gene_to_protein[gene].add(v['protein_id'])
        gene_to_protein[v['protein_id']].add(v['protein_id'])
    mapping={g:next(iter(v)) for g,v in gene_to_protein.items() if len(v)==1}
    libraries={}; metadata={}; actual_exposure=defaultdict(set)
    for cell in CELLS:
        p=DATA/f'{cell}_rows.csv';sources.append(record(p))
        rows=pd.read_csv(p,keep_default_na=False);rows=rows.loc[rows.pert_type.eq('trt_cp')]
        metadata[cell]=rows
        for parent,g in rows.groupby('parent_key'):
            for split in g.split.unique():actual_exposure[parent].add(f'{cell}:{split}')
        p=DATA/f'{cell}.npz';sources.append(record(p))
        with np.load(p,allow_pickle=False) as z:libraries[cell]=set(z['proteins'].astype(str))
    coverage=[];cellrows=[];measurementrows=[];mappingrows=[]
    for gene in matrix.index:
        mappingrows.append({'source_gene':gene,'protein_id':mapping.get(gene,''),
                            'status':'unique_canonical_match' if gene in mapping else 'unmapped_or_ambiguous',
                            'direct_kinobead_binder':gene in annotations.index and annotations.loc[gene,'DirectBinder']=='x'})
    for i,row in compounds.iterrows():
        name=row['Compound'];exact,parent,scaffold=structure(row.SMILES)
        matches=lincs.get(parent,[]);pert_ids={v[0] for v in matches}
        finite=[]
        if name in matrix.columns:
            for gene,value in matrix[name].items():
                if value=='':continue
                try:value=float(value)
                except ValueError:continue
                if not np.isfinite(value) or value<=0:continue
                direct=gene in annotations.index and annotations.loc[gene,'DirectBinder']=='x'
                finite.append((gene,mapping.get(gene,''),value,direct))
                measurementrows.append({'compound':name,'source_sheet':'Kinobeads Drugmatrix - all',
                                        'source_row':int(matrix.index.get_loc(gene))+2,'source_column':int(matrix.columns.get_loc(name))+2,
                                        'gene':gene,'protein_id':mapping.get(gene,''),'kd_app_nm':value,
                                        'direct_kinobead_binder':direct,'submicromolar':value<1000})
        cells=[];positivecells=[];availabletargets=set()
        for cell in CELLS:
            eligible=metadata[cell].loc[metadata[cell].pert_id.isin(pert_ids)]
            if eligible.empty:continue
            cells.append(cell)
            targets={p for g,p,v,direct in finite if p in libraries[cell] and direct}
            positives={p for g,p,v,direct in finite if p in libraries[cell] and direct and v<1000}
            availabletargets|=targets
            if positives:positivecells.append(cell)
            cellrows.append({'compound':name,'parent_key':parent,'cell':cell,'signatures':len(eligible),
                             'lincs_pert_ids':'|'.join(sorted(eligible.pert_id.unique())),
                             'prior_g2cp_split':'|'.join(sorted(eligible.split.unique())),
                             'reported_direct_targets_in_library':len(targets),'submicromolar_direct_targets_in_library':len(positives),
                             'mapped_targets':'|'.join(sorted(targets)), 'blank_measurement_policy':'unknown, not negative'})
        history=actual_exposure.get(parent,set())
        train_or_val=any(s.endswith(':train') or s.endswith(':val') for s in history)
        base={'compound':name,'source_sheet':'Tool compounds','source_row':i+2,'smiles':row.SMILES,
              'canonical_identity':exact,'parent_key':parent,'valid_structure':bool(parent),
              'source_chemotype':row.Chemotype,'lincs_metadata_match':bool(matches),
              'lincs_exact_match':any(e==exact for _,e in matches),'lincs_pert_ids':'|'.join(sorted(pert_ids)),
              'kiba_seen':parent in training['kiba'],'davis_seen':parent in training['davis'],
              'kiba_scaffold_seen':scaffold in scaffolds['kiba'],'davis_scaffold_seen':scaffold in scaffolds['davis'],
              'old_results5_59':parent in old59,'old_results5_68':parent in old68,'old_braf':parent in braf,
              'old_nanobret':parent in nano,'g2cp_global_split':split_lookup.get(parent,''),
              'g2cp_actual_expression_exposure':'|'.join(sorted(history)),
              'g2cp_train_or_val_expression_seen':train_or_val,'eligible_cells':'|'.join(cells),
              'cells_with_reported_positive':'|'.join(positivecells),'finite_reported_targets':len(finite),
              'direct_targets_in_any_library':len(availabletargets)}
        base['fresh_metadata_candidate']=bool(parent and cells and parent not in training['kiba'] and parent not in training['davis']
            and parent not in old68 and parent not in braf and parent not in nano)
        base['strict_unexposed_candidate']=base['fresh_metadata_candidate'] and not train_or_val
        base['strict_with_positive_coverage']=base['strict_unexposed_candidate'] and bool(positivecells)
        coverage.append(base)
    frame=pd.DataFrame(coverage)
    outputs={'external_compound_coverage.csv':frame,'external_cell_coverage.csv':pd.DataFrame(cellrows),
             'external_reported_measurements.csv':pd.DataFrame(measurementrows),'external_target_mapping.csv':pd.DataFrame(mappingrows)}
    for name,df in outputs.items():write_once(DEST/name,df.to_csv(index=False))
    counts={key:int(frame[key].sum()) for key in ['valid_structure','lincs_metadata_match','lincs_exact_match',
            'kiba_seen','davis_seen','old_results5_59','old_results5_68','old_braf','old_nanobret',
            'fresh_metadata_candidate','strict_unexposed_candidate','strict_with_positive_coverage']}
    counts['source_compounds']=len(frame);counts['local_eligible_response_compounds']=int(frame.eligible_cells.ne('').sum())
    counts['response_and_positive_compounds']=int(frame.cells_with_reported_positive.ne('').sum())
    save_json(DEST/'external_coverage_audit.json',{'source':'Reinecke et al, doi:10.1038/s41589-023-01459-3',
              'counts':counts,'strict_candidate_names':frame.loc[frame.strict_unexposed_candidate,'compound'].tolist(),
              'external_model_scores_computed':False,'new_expression_values_loaded':False,
              'conditions':'Existing prepared QC-passing chemical rows: 10uM/24h, five cells; frozen genetic TRAIN library 96h',
              'limitations':['Coverage is source-row counts; parent identity is the inference unit.',
                            'Blank affinity cells do not establish measured negatives; matrix contains identified target affinities.',
                            'DirectBinder annotates direct binding to Kinobeads; not a causal drug-mechanism label.',
                            'Parent mapping ignores stereochemistry in connectivity key; exact match also retained.',
                            'Upstream pretrained encoder exposure and full affinity-source lineage remain unresolved.',
                            'No confirmation cohort is frozen or scored in this audit.'],
              'inputs':sources,'outputs':[record(DEST/n) for n in outputs]},immutable=True)
    print(json.dumps(counts,indent=2));print(frame.loc[frame.fresh_metadata_candidate,['compound','eligible_cells','cells_with_reported_positive','g2cp_global_split']].to_string(index=False))


if __name__=='__main__':main()
