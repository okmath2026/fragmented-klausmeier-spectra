"""Register the strict signed-coefficient and anchor-dependent cubic studies."""
import csv,json
from spectral_study import ROOT,DATA
from signed_second_study import main as signed_main,EXPERIMENT_NAMES
from cubic_anchor_study import main as cubic_main


def main():
    signed=signed_main();cubic=cubic_main()
    names=(*EXPERIMENT_NAMES,'cubic_anchor');experiments={}
    for name in names:
        with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
            experiments[name]=list(csv.DictReader(stream))
    path=ROOT/'output/pdf/shape_spectral_results.csv'
    with path.open(newline='',encoding='utf-8') as stream:
        combined=[row for row in csv.DictReader(stream) if row['experiment'] not in names]
    for name,rows in experiments.items():combined.extend(dict(experiment=name,**row) for row in rows)
    fields=list(dict.fromkeys(k for row in combined for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    summary=json.loads((DATA/'summary.json').read_text())
    summary.update(experiment_rows=len(combined),numerical_experiment_count=len({r['experiment'] for r in combined}),
        narrow_signed_second_checks=signed['checks'],cubic_anchor_checks=cubic['checks'],
        signed_second_coefficient_certified=True,finite_ratio_localization=False)
    for name,rows in experiments.items():summary[name+'_rows']=len(rows)
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(result_rows=len(combined),numerical_experiments=summary['numerical_experiment_count'],
        new_experiments={k:len(v) for k,v in experiments.items()}),indent=2),flush=True)
    from research_refinement_study import main as research_refinements
    research_refinements()


if __name__=='__main__':main()
