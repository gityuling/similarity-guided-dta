#!/usr/bin/env python3
"""Inventory measured channels without interpreting classifier negatives as affinity bounds."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'reports/bib_selectivity_20261021'


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main():
    manifest = json.loads((OUT/'download_manifest.json').read_text())
    assert manifest['failed'] == 0 and manifest['verified'] == manifest['total']
    with (OUT/'target_coverage.csv').open() as f:
        genes = {r['gene'] for r in csv.DictReader(f)}
    fields = ['compound', 'gene', 'state', 'source_row', 'picked_intensity',
              'dmso_intensity', 'residual_100nm', 'residual_1000nm', 'dmso_variance',
              'variance_100nm', 'variance_1000nm', 'target_probability',
              'target_annotation', 'source_kd', 'correction_factor']
    totals, codes, channels = Counter(), Counter(), Counter()
    target_counts = defaultdict(Counter)
    file_summaries = []
    reference = defaultdict(dict)
    reference_path = OUT.parent/'bib_mechanism_20261021/sources/reinecke_matrix.csv'
    with reference_path.open() as f:
        for row in csv.DictReader(f):
            gene = row.pop('Gene Name')
            if gene in genes:
                for compound, value in row.items():
                    value = finite(value)
                    if value is not None and value > 0:
                        reference[compound][gene] = value
    reconciliation = []
    with (OUT/'measurement_states.csv').open('w',newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in manifest['files']:
            path = OUT/'sources'/item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
            exact = defaultdict(list)
            grouped = set()
            for index, row in enumerate(rows, start=2):
                codes[row['Target.annotation']] += 1
                channels[row['Picked.intensity']] += 1
                gene = row['Gene.names']
                if gene in genes:
                    exact[gene].append((index,row))
                else:
                    grouped.update(set(gene.split(';')) & genes)
            counts = Counter()
            compound = item['file'].removesuffix('_final.csv')
            for gene in sorted(genes):
                record = dict.fromkeys(fields, '')
                record.update(compound=compound,gene=gene)
                if len(exact[gene]) > 1 or (gene in grouped and exact[gene]):
                    state = 'ambiguous_duplicate_or_group'
                elif not exact[gene]:
                    state = 'group_only' if gene in grouped else 'not_reported'
                else:
                    index,row = exact[gene][0]
                    channel = row['Picked.intensity']
                    record.update(source_row=index,picked_intensity=channel,
                        target_probability=row['Target.probability'],target_annotation=row['Target.annotation'],
                        source_kd=row['Kd'],correction_factor=row['CorrectionFactor'])
                    if channel not in ['LFQ.intensity','Intensity']:
                        state = 'unrecognized_channel'
                    else:
                        keys = {'dmso_intensity':'Median.DMSO.'+channel,
                            'residual_100nm':'Median.Normalized.'+channel+'.100nM',
                            'residual_1000nm':'Median.Normalized.'+channel+'.1000nM',
                            'dmso_variance':'Var.Normalized.'+channel+'.DMSO',
                            'variance_100nm':'Var.Normalized.'+channel+'.100nM',
                            'variance_1000nm':'Var.Normalized.'+channel+'.1000nM'}
                        values = {k:finite(row.get(v)) for k,v in keys.items()}
                        record.update({k:row.get(v,'') for k,v in keys.items()})
                        if any(v is None for v in values.values()):
                            state = 'incomplete_selected_channel'
                        elif values['dmso_intensity'] <= 0 or any(v < 0 for k,v in values.items() if k!='dmso_intensity'):
                            state = 'invalid_selected_channel'
                        else:
                            state = 'finite_signal_not_qc_certified'
                record['state'] = state
                writer.writerow(record)
                totals[state]+=1; counts[state]+=1; target_counts[gene][state]+=1
                if gene in reference[compound]:
                    expected = reference[compound][gene]
                    actual = finite(record['source_kd'])
                    status = 'unresolved_mapping_or_missing'
                    if actual is not None:
                        status = 'agrees' if math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-8) else 'numerical_discrepancy'
                    reconciliation.append({'compound':compound,'gene':gene,'table_kd':expected,
                        'processed_kd':actual,'status':status,'source_row':record['source_row']})
            file_summaries.append(dict(compound=compound, source_rows=len(rows), **counts))
    with (OUT/'processed_target_coverage.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['gene']+sorted(totals))
        writer.writeheader()
        for gene in sorted(genes):
            writer.writerow({'gene':gene,**{s:target_counts[gene][s] for s in sorted(totals)}})
    report = {'files':len(manifest['files']), 'modeled_genes':len(genes),
        'states':dict(totals),'annotation_codes_all_source_rows':dict(codes),
        'picked_channels_all_source_rows':dict(channels),
        'file_summaries':file_summaries,
        'interpretation':'Finite selected-channel measurements are NOT QC-certified negatives or affinity bounds. No selection-model scoring performed.'}
    with (OUT/'affinity_source_reconciliation.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['compound','gene','table_kd','processed_kd','status','source_row'])
        writer.writeheader();writer.writerows(reconciliation)
    report['affinity_reconciliation'] = dict(Counter(r['status'] for r in reconciliation))
    assert sum(totals.values()) == len(genes)*len(manifest['files'])
    (OUT/'measurement_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='file_summaries'},indent=2))


if __name__ == '__main__':
    main()
