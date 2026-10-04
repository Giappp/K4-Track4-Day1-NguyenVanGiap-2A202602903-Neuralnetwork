"""Check artifacts agree with saved histories and official evaluator output."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
import openpyxl
from results_table import load_results


def main():
    out=Path(__file__).resolve().parent.parent
    results=load_results(out/'results')
    wb=openpyxl.load_workbook(out/'experiments.xlsx',data_only=True)
    assert wb.sheetnames==['Legend','Experiments','Seeds','Summary']
    rows=list(wb['Experiments'].values)
    headers=rows[0]
    rows=[dict(zip(headers,r)) for r in rows[1:] if r[0]]
    assert len(rows)==len(results)>0
    assert len({r['exp_id'] for r in rows})==len(rows)
    indexed={r['exp_id']:r for r in rows}
    for r in results:
        exp=r['cfg']['exp_id'];row=indexed[exp]
        assert (out/'figures'/f'{exp}.png').exists()
        assert row['figure_file']==f'figures/{exp}.png'
        assert len(r['history']['epoch'])==r['cfg']['epochs']
        for key in ('step0_loss','val_acc','val_macro_f1','best_val_loss','final_train_loss','final_val_loss'):
            assert np.isclose(row[key],r['summary'][key],atol=1e-12), (exp,key)
    figure_ids={p.stem for p in (out/'figures').glob('*.png') if not p.stem.startswith(('compare_','health_','eval_'))}
    assert figure_ids==set(indexed)
    choice=json.loads((out/'results/selection.json').read_text())
    final=json.loads((out/'eval_result.json').read_text())
    baseline=json.loads((out/'eval_baseline.json').read_text())
    expected=max((r for r in results if r['cfg']['seed']==1),key=lambda r:r['summary']['val_macro_f1'])
    assert choice['exp_id']==expected['cfg']['exp_id']
    for exp,scores in (('base-s1',baseline),(choice['exp_id'],final)):
        assert np.isclose(indexed[exp]['eval_macro_f1'],scores['macro_f1'],atol=1e-12)
        assert np.isclose(indexed[exp]['eval_acc'],scores['accuracy'],atol=1e-12)
    assert all(r['eval_acc'] is None and r['eval_macro_f1'] is None for r in rows if r['exp_id'] not in ('base-s1',choice['exp_id']))
    predictions=pd.read_csv(out/'predictions_eval.csv')
    assert list(predictions)==['row_id','pred'] and len(predictions)==116203
    assert predictions.row_id.is_unique and predictions.pred.between(0,6).all()
    nb=nbformat.read(out/'code/lab.ipynb',as_version=4);nbformat.validate(nb)
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
    report=(out/'REPORT.md').read_text()
    assert f"{final['macro_f1']:.6f}" in report and f"{final['accuracy']:.6f}" in report
    for sheet in wb:
        assert not any(c.data_type=='e' for row in sheet for c in row)
    assert not list(out.rglob('*.pt')) and not list(out.rglob('*.pth'))
    print(f'PASS: {len(results)} runs, matching figures/workbook/histories; notebook executed; selection and eval agree.')

if __name__=='__main__':main()
