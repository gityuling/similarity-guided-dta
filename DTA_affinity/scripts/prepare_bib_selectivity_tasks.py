#!/usr/bin/env python3
"""Define all same-KLIFS-family directed pairs, independent of model efficacy."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bib_selectivity_20261021'


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    annotation=OUT/'sources/klifs_human_information.json'
    records=json.loads(annotation.read_text());index=defaultdict(list)
    for r in records:
        if r['species'].lower()=='human':index[r['uniprot']].append(r)
    mappings=[];families=defaultdict(list)
    for r in read(OUT/'target_coverage.csv'):
        matches=index[r['protein_id']]
        status='unique' if len(matches)==1 else ('unmapped' if not matches else 'multiple_kinase_domains')
        item=dict(gene=r['gene'],protein_id=r['protein_id'],status=status,family='',kinase_group='',klifs_id='')
        if status=='unique':
            k=matches[0];item.update(family=k['family'],kinase_group=k['group'],klifs_id=k['kinase_ID'])
            if k['family'] and k['family'].lower() not in ['other','unique','unclassified']:
                families[k['family']].append(r['gene'])
        mappings.append(item)
    save('klifs_mapping_audit.csv',mappings)
    label_development={r['compound'] for r in read(OUT/'full_dose_label_comparison.csv')}
    compounds=[r for r in read(OUT/'compound_audit.csv') if r['primary_candidate']=='True']
    for r in compounds:r['label_validation_used']=r['compound'] in label_development
    evaluation=[r for r in compounds if not r['label_validation_used']]
    save('task_candidate_pool.csv',compounds)
    names={r['compound'] for r in evaluation}
    qc=defaultdict(dict)
    for r in read(OUT/'qc_measurement_states.csv'):
        if r['profile']=='reference' and r['compound'] in names:
            qc[r['gene']][r['compound']]=r['state']
    tasks=[]
    for family,genes in sorted(families.items()):
        for a in sorted(genes):
            for b in sorted(genes):
                if a==b:continue
                coverage=sum(qc[a][c]!='unknown' and qc[b][c]!='unknown' for c in names)
                tasks.append(dict(task_id=a+'__avoid__'+b,family=family,target=a,avoid=b,
                    candidate_pool_size=len(names),joint_qc_coverage=coverage,eligible=coverage>=50))
    save('family_tasks.csv',tasks)
    used_genes={r[g] for r in tasks if r['eligible'] for g in ['target','avoid']}
    proteins={r['protein_id']:r['Sequence'] for r in read(ROOT/'reports/bib_nanobret_20261018/protein_annotation_audit.csv') if r['canonical_sequence_matches'].lower()=='true'}
    gene_ids={r['gene']:r['protein_id'] for r in mappings}
    queries=[]
    for compound in sorted(evaluation,key=lambda r:r['compound']):
        for gene in sorted(used_genes):
            protein=gene_ids[gene]
            queries.append(dict(row_id='sel_'+str(len(queries)),compound=compound['compound'],
                drug_id=compound['parent_key'],protein_id=protein,gene=gene,
                Canonical_SMILES=compound['canonical_identity'],Sequence=proteins[protein],eligible_folds='0|1|2|3|4'))
    save('query_table.csv',queries)
    report={'task_definition':'Every directed pair within the same KLIFS family; unique UniProt/domain mapping; >=50 joint reference-QC candidates. No success-count eligibility.',
        'source_url':'https://klifs.net/api/kinase_information?species=HUMAN',
        'klifs_records':len(records),'mapped_unique_genes':sum(r['status']=='unique' for r in mappings),
        'multiple_domain_genes':[r['gene'] for r in mappings if r['status']=='multiple_kinase_domains'],
        'unmapped_genes':[r['gene'] for r in mappings if r['status']=='unmapped'],
        'label_development_compounds':sorted(label_development),'evaluation_compounds':len(names),
        'defined_tasks':len(tasks),'coverage_eligible_tasks':sum(r['eligible'] for r in tasks),
        'families':len({r['family'] for r in tasks if r['eligible']}),
        'query_genes':len(used_genes),'query_rows':len(queries),'model_scores_loaded':False,
        'status':'Task membership fixed for exploratory evaluation; scoring protocol and fair DTI baseline not yet frozen.',
        'limitations':['Same-family membership does not establish equivalent function or full kinome selectivity.',
            'This source was used in QC development; removing four curve-validation compounds does not restore untouched external confirmation.',
            'No broad weak-competition validation cohort is available from the checked full-dose sheets.'],
        'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [annotation,OUT/'family_tasks.csv',OUT/'task_candidate_pool.csv',OUT/'query_table.csv',Path(__file__)]}}
    (OUT/'task_definition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2))


if __name__=='__main__':main()
