"""Rebuild and register second-order and strict-transition extensions."""
import csv,json
from spectral_study import ROOT,DATA
from transition_study import main as transition_main,EXPERIMENT_NAMES
from second_order_study import main as second_main


def main():
    transitions=transition_main()
    second=second_main()
    names=(*EXPERIMENT_NAMES,*second['experiments'])
    experiments={}
    for name in names:
        with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
            experiments[name]=list(csv.DictReader(stream))
    path=ROOT/'output/pdf/shape_spectral_results.csv'
    with path.open(newline='',encoding='utf-8') as stream:
        combined=[row for row in csv.DictReader(stream) if row['experiment'] not in experiments]
    for name,rows in experiments.items():
        combined.extend(dict(experiment=name,**row) for row in rows)
    fields=list(dict.fromkeys(k for row in combined for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    summary=json.loads((DATA/'summary.json').read_text())
    summary.update(experiment_rows=len(combined),numerical_experiment_count=len({r['experiment'] for r in combined}),
        transition_study=transitions,second_order_checks=second['checks'],
        second_budget_checks=second['checks']['second_budget_checks'])
    for name,rows in experiments.items():summary[name+'_rows']=len(rows)
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(new_experiments={k:len(v) for k,v in experiments.items()},
        result_rows=len(combined),experiments=summary['numerical_experiment_count']),indent=2),flush=True)
    from certificate_refinement_study import main as certificate_refinement_main
    certificate_refinement_main()


if __name__=='__main__':main()
