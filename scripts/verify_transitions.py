"""Standard-library replay of strict local all-integer count transitions."""
import csv,json
from pathlib import Path
from fractions import Fraction
from validated_fragment_principal import fragment_principal_certificate,fragment_principal_self_check
from validated_water import rational_water_certificate
from validated_peaks import rayleigh_envelope_certificate
from validated_coupled_optimum import coupled_tail_certificate
from validated_transitions import global_transition_certificate,transition_self_check

if not __debug__:raise RuntimeError('Run validated certificates without -O.')
ROOT=Path(__file__).resolve().parents[1]
CASES=[('4',(79,80,81),('11.9','12.15'),40,240),
       ('6',(69,70,71),('11.8','12.1'),50,200)]


def _rows(name):
    with (ROOT/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _json(name):return json.loads((ROOT/'data'/(name+'.json')).read_text())


def _key(row):return (str(row['D']),int(row['m']),int(row['features']))


def _compare(actual,saved,label):
    for key,value in actual.items():
        assert key in saved,(label,'missing field',key)
        assert str(value)==saved[key],(label,key,str(value),saved[key])


def _independent_fraction_checks(proof,tail,counts,window,audit):
    """Recheck phase inequalities independently from exact primitive endpoints."""
    bym={int(row['m']):row for row in proof};L,U=map(Fraction,window)
    def f(row,r,lower):
        return Fraction(row['beta_lower' if lower else 'beta_upper'])-r*Fraction(row['q_upper' if lower else 'q_lower'])
    independent_pairs=[]
    for left,right in zip(counts,counts[1:]):
        a,b=bym[left],bym[right]
        numeratorL=Fraction(a['beta_lower'])-Fraction(b['beta_upper'])
        numeratorU=Fraction(a['beta_upper'])-Fraction(b['beta_lower'])
        denominatorL=Fraction(a['q_lower'])-Fraction(b['q_upper'])
        denominatorU=Fraction(a['q_upper'])-Fraction(b['q_lower'])
        assert 0<numeratorL<=numeratorU and 0<denominatorL<=denominatorU
        lo,hi=numeratorL/denominatorU,numeratorU/denominatorL
        independent_pairs.append((lo,hi))
    assert L<independent_pairs[0][0]<=independent_pairs[0][1]<independent_pairs[1][0]<=independent_pairs[1][1]<U
    for pair,(lo,hi) in zip(audit['pairs'],independent_pairs):
        assert Fraction(pair['crossing_lower_exact'])==lo
        assert Fraction(pair['crossing_upper_exact'])==hi
        assert Fraction(pair['crossing_lower'])<=lo<=hi<=Fraction(pair['crossing_upper'])
        assert Fraction(pair['crossing_width_upper'])>=hi-lo
    small={r['m']:r for r in audit['smaller_competitors']}
    large={r['m']:r for r in audit['larger_competitors']}
    assert set(small)==set(range(2,counts[0]))
    assert set(large)==set(range(counts[-1]+1,int(tail['anchor_m'])))
    for m,row in bym.items():
        if m<counts[0]:
            gap=f(bym[counts[0]],L,True)-f(row,L,False)
            assert gap==Fraction(small[m]['gap_exact'])>0
            assert 0<Fraction(small[m]['gap_lower'])<=gap
        if m>counts[-1]:
            gap=f(bym[counts[-1]],U,True)-f(row,U,False)
            assert gap==Fraction(large[m]['gap_exact'])>0
            assert 0<Fraction(large[m]['gap_lower'])<=gap
    tailgap=f(bym[counts[-1]],U,True)-Fraction(tail['all_m_ge_anchor_beta_upper'])
    assert tailgap==Fraction(audit['infinite_tail']['gap_exact'])>0
    assert 0<Fraction(audit['infinite_tail']['gap_lower'])<=tailgap


def main():
    checks=_json('transition_checks')
    saved_candidates=_rows('transition_candidates');saved_pairs=_rows('transition_crossings')
    saved_global=_rows('transition_global');new_vectors=_json('transition_vectors')
    new_audits={_key(a['summary']):a for a in _json('transition_audits')}
    assert len(saved_candidates)==len(new_vectors)==len(new_audits)==6
    assert len(saved_pairs)==4 and len(saved_global)==2
    expected={(D,m,p) for D,counts,window,p,M in CASES for m in counts}
    assert {_key(r) for r in saved_candidates}=={_key(v) for v in new_vectors}==set(new_audits)==expected
    old_rows=_rows('coupled_candidates')+_rows('coupled_refinement')+_rows('coupled_competitors')
    old_lookup={_key(r):r for r in old_rows}
    new_lookup={_key(r):r for r in saved_candidates}
    old_vectors={_key(v):v for v in _json('coupled_peak_vectors')}
    new_vectors={_key(v):v for v in new_vectors}
    old_audits={_key(a['summary']):a for a in _json('coupled_peak_audits')}
    saved_global_audits={a['summary']['D']:a for a in _json('transition_global_audits')}
    assert set(saved_global_audits)=={'4','6'}
    envelope_vectors=_json('peak_envelope_vectors')
    saved_envelopes=_rows('peak_envelopes');saved_tails=_rows('coupled_tails')
    replay_count=0;new_checks=[];global_checks=[];replayed_pairs=[]
    for D,counts,window,p,M in CASES:
        entries=checks['selection'][D]
        assert checks['finite_coverage'][D]==[2,M-1] and checks['windows'][D]==list(window)
        assert len(entries)==M-2 and {r['m'] for r in entries}==set(range(2,M))
        proof=[]
        for entry in entries:
            key=(D,entry['m'],entry['features'])
            source=entry['source'];assert source in ('transition','coupled')
            assert (source=='transition')==(entry['m'] in counts)
            saved=(new_lookup if source=='transition' else old_lookup)[key]
            vector=(new_vectors if source=='transition' else old_vectors)[key]
            expected_audit=(new_audits if source=='transition' else old_audits)[key]
            record,audit=fragment_principal_certificate(D,entry['m'],entry['features'],vector['q'],60,'2')
            assert audit==expected_audit,('complete principal audit',key)
            _compare(record,saved,('principal endpoints',key))
            water=rational_water_certificate(Fraction(2,entry['m']),Fraction(1))
            assert water['q_lower']==saved['q_lower'] and water['q_upper']==saved['q_upper']
            row=dict(record,dw='1',q_lower=water['q_lower'],q_upper=water['q_upper'])
            proof.append(row);replay_count+=1
            if source=='transition':
                assert entry['features']==p
                new_checks.append(fragment_principal_self_check(D,entry['m'],p,vector['q']))
        vector=next(v for v in envelope_vectors if v['D']==D and int(v['N'])==640)
        envelope=rayleigh_envelope_certificate(D,vector['N'],vector['q'],vector['precision'],vector.get('V','2'))
        _compare(envelope,next(r for r in saved_envelopes if r['D']==D),('Rayleigh envelope',D))
        tail=coupled_tail_certificate(envelope,M)
        _compare(tail,next(r for r in saved_tails if r['D']==D),('infinite tail',D))
        summary,audit=global_transition_certificate(proof,tail,counts,window)
        _compare(summary,next(r for r in saved_global if r['D']==D),('global phase diagram',D))
        assert audit==saved_global_audits[D],('full phase audit',D)
        _independent_fraction_checks(proof,tail,counts,window,audit)
        replayed_pairs.extend(audit['pairs'])
        global_checks.append(transition_self_check(proof,tail,counts,window))
        print(f'transition proof D={D}: {M-2} finite counts and every m>={M} reconstructed',flush=True)
    for actual in replayed_pairs:
        saved=next(r for r in saved_pairs if (r['D'],int(r['left_m']))==(actual['D'],actual['left_m']))
        _compare(actual,saved,('pair crossing',actual['D'],actual['left_m']))
    assert new_checks==checks['candidate_checks']
    assert global_checks==checks['global_checks']
    # Explicitly test rejection of an incomplete all-count range.
    try:global_transition_certificate(proof[:-1],tail,counts,window)
    except ValueError:pass
    else:raise AssertionError('incomplete finite coverage was accepted')
    print(json.dumps(dict(reconstructed_principal_records=replay_count,new_candidate_records=6,
        exact_crossing_brackets=4,all_count_global_phase_diagrams=2,
        complete_saved_audits_reproduced=True,independent_fraction_checks=True,
        precision_and_context_checks=True,incomplete_coverage_rejected=True,
        dependencies='Python standard library only; exact saved trial inputs'),indent=2))


if __name__=='__main__':main()
