"""Generate reproducible uniform Hilbert-space cubic-remainder data and table."""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
import csv
import json

from validated_hilbert_remainder import hilbert_remainder_family

ROOT = Path(__file__).resolve().parent.parent


def selected_sources():
    saved = json.loads((ROOT/'data/signed_second_vectors.json').read_text(encoding='utf-8'))
    selected = []
    for D, p in [('4', 40), ('6', 50)]:
        source = next(r for r in saved if r['D'] == D and r['features'] == p)
        selected.append(dict(D=D, V=source.get('V', '2'), p=p,
                             trial=source['trial']['spectral']))
    return selected


def _rounded(value, places, upper=True):
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_CEILING if upper else ROUND_FLOOR
        return format(Decimal(value).quantize(Decimal(1).scaleb(-places)), 'f')


def main():
    rows = []
    minima = []
    for source in selected_sources():
        family, audit = hilbert_remainder_family(source['D'], source['p'], source['trial'],
            (70, 80, 160, 320), 60, source['V'])
        for record in family:
            record['minimum_half_anchor'] = audit['minimum_half_anchor']
        rows.extend(family)
        minima.append((source['D'], audit['minimum_half_anchor']))
        print(json.dumps(dict(D=source['D'], features=source['p'],
            minimum_half_anchor=audit['minimum_half_anchor'],
            target=[{k:r[k] for k in ('M','eta_upper','K3_upper','anchor_error_upper')}
                    for r in family if r['M'] in (70,80)])))
    path = ROOT/'data/hilbert_remainders.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    output = [r'\begin{table}[tbp]', r'\centering\small',
        r'\caption{Uniform cubic remainder bounds in the Gaussian feature space. Each row holds for every integer $m\geq M$. The smallest thresholds are certified for the working condition $\eta_M^H\leq1/2$. The norm estimates remain too conservative for tight count localisation at a fixed ratio.}',
        r'\label{tab:hilbert-remainders}', r'\begin{tabular}{rrrrrr}', r'\toprule',
        r'$D$ & $M$ & $\gamma_{\rm e}^-$ & $\eta_M^{H,+}$ & $K_{3,H}^+$ & $K_{3,H}^+/M^3$\\',
        r'\midrule']
    for r in rows:
        if r['M'] not in (r['minimum_half_anchor'], 70, 80, 160):
            continue
        output.append(f"{r['D']} & {r['M']} & {_rounded(r['gamma_lower'],8,False)} & "
            f"{_rounded(r['eta_upper'],6)} & {_rounded(r['K3_upper'],2)} & "
            f"{_rounded(r['anchor_error_upper'],7)}"+r'\\')
    output.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'generated/hilbert_remainders.tex').write_text('\n'.join(output)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(rows), data=str(path), minimum_anchors=minima)))


if __name__ == '__main__':
    main()
