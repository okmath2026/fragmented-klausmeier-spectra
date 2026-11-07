"""Regenerate exact anchor-dependent cubic budgets, without global mutations."""
import csv,json
from pathlib import Path
from validated_cubic_anchor import (minimum_cubic_anchor,cubic_anchor_certificate,
    cubic_anchor_self_check,cubic_anchor_search_audit)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
EXPERIMENT_NAMES=('cubic_anchor',)


def main(write_table=True):
    with (DATA/'second_budgets.csv').open(newline='',encoding='utf-8') as stream:
        sources=list(csv.DictReader(stream))
    assert len(sources)==2 and {row['D'] for row in sources}=={'4','6'}
    rows=[];checks=[];searches=[]
    for record in sources:
        M=minimum_cubic_anchor(record)
        for factor in (1,2,4,8):
            rows.append(dict(**cubic_anchor_certificate(record,M*factor),anchor_multiplier=factor))
        checks.append(cubic_anchor_self_check(record))
        searches.append(cubic_anchor_search_audit(record))
    with (DATA/'cubic_anchor.csv').open('w',newline='',encoding='ascii') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit=dict(experiment_rows=len(rows),anchor_factors=[1,2,4,8],checks=checks,anchor_searches=searches,
        scope='certified C3 norm-majorant reduction; not practical finite-ratio localisation')
    (DATA/'cubic_anchor_checks.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='ascii')
    if write_table:
        from spectral_study import table,decimal_display,scientific_upper
        table('cubic_anchor',
            'Cubic remainder bounds at different thresholds. Each row holds for every integer $m\\geq M$. The gain bounds the ratio of the old norm estimate to the new one from below. It does not measure a ratio of actual spectral errors.',
            ['$D$','$M$','$\\eta_M\\leq$','$K_3(M)\\leq$','Budget gain $\\geq$'],
            [[r['D'],str(r['M']),decimal_display(r['eta_upper'],5,True),
              scientific_upper(r['K3_upper']),decimal_display(r['K3_reduction_lower'],0)] for r in rows])
    result=dict(experiment_rows={'cubic_anchor':len(rows)},checks=checks,
        minimum_anchors={r['D']:r['minimum_M'] for r in checks})
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':main()
