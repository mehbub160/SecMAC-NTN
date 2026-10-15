"""Run the experiments of the paper.

    python3 run_experiments.py                 # everything (about 1 h on 2 cores)
    python3 run_experiments.py mac_density     # one experiment
    python3 run_experiments.py --scale 0.2 poison   # a fifth of the replications

Every experiment is a list of (point, config) pairs; replication r of a point
uses seed base + r, and all schemes at the same point share the seed, hence
the topology, clocks and traffic.  Raw per-run results go to
results/raw/<experiment>.jsonl; scripts/analyze.py turns them into tables.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secmac.sim import Simulator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'results', 'raw')
CFG = json.load(open(os.path.join(HERE, 'configs', 'default.json')))
SEEDS = json.load(open(os.path.join(HERE, 'configs', 'seeds.json')))

RUN = dict(T_warm=600.0, T_meas=3600.0, T_tail=600.0)
RUN_SHORT = dict(T_warm=600.0, T_meas=1800.0, T_tail=600.0)

SCHEMES = ['pure', 'slotted', 'reservation', 'tdma', 'secmac']
POLICIES = ['counter', 'static', 'cap', 'tvw']


def variant(name):
    """SecMAC-NTN ablation variants."""
    v = dict(scheme='secmac', policy='tvw')
    if name == 'full':
        return v
    if name == 'no_trust':
        v['secmac'] = dict(trust_aware=False)
    elif name == 'no_stability':
        v['secmac'] = dict(stability=False)
        v['casam'] = dict(n_d=1, hyst=0.0)
    elif name == 'no_tvw':
        v['policy'] = 'counter'
    elif name == 'no_admission':
        v['secmac'] = dict(admission=False)
    elif name == 'slotted_casam':
        v['scheme'] = 'slotted'
    elif name == 'pure_casam':
        v['scheme'] = 'pure'
    else:
        raise ValueError(name)
    return v


def experiments():
    E = {}
    # attack-free statistics of the trust inputs, used to set psi_G (separate seeds)
    E['calibration'] = dict(reps=6, points=[
        (dict(N=N), dict(N=N, scheme='secmac', trace=True, **RUN)) for N in (300, 600, 1200, 2000)])
    # access performance against density
    E['mac_density'] = dict(reps=20, points=[
        (dict(N=N, scheme=s), dict(N=N, scheme=s, **RUN))
        for N in (100, 200, 400, 600, 800, 1200, 1600, 2000) for s in SCHEMES])
    E['reservation_tuned'] = dict(reps=10, points=[
        (dict(N=N, TW=tw, n_grant=g), dict(N=N, scheme='reservation', TW=tw, n_grant=g, **RUN))
        for N in (400, 1200) for tw, g in ((20, 10),)])
    # NTN path delay (scale 45 gives GEO-like one-way delays)
    E['mac_delay'] = dict(reps=10, points=[
        (dict(scale=sc, scheme=s), dict(N=600, scheme=s, ntn_delay_scale=sc, **RUN))
        for sc in (1.0, 5.0, 15.0, 45.0) for s in ('pure', 'slotted', 'tdma', 'secmac')])
    # replay: suppress-and-replay (A2) with log-uniform delay, duplicates (A1)
    for cfgname, extra in (('pure', dict(secmac=dict(mode_select=False))), ('amc', {})):
        E['replay_delay_' + cfgname] = dict(reps=20, points=[
            (dict(policy=p), dict(N=300, scheme='secmac', policy=p, a2_rate=0.1, a1_rate=0.1, **extra, **RUN))
            for p in POLICIES])
    # earliest replay (5 ms after the end of the frame) against the synchronisation
    E['replay_sync'] = dict(reps=20, points=[
        (dict(T_sync=ts, eps_ts=e, policy=p),
         dict(N=300, scheme='secmac', secmac=dict(mode_select=False), policy=p, a2_rate=0.1,
              replay_delay=('fixed', 0.005), T_sync=ts, eps_ts=e, **RUN))
        for ts, e in ((3600.0, 0.1), (6 * 3600.0, 0.1), (24 * 3600.0, 0.1), (6 * 3600.0, 0.01))
        for p in ('static', 'cap', 'tvw')])
    E['replay_ntn'] = dict(reps=10, points=[
        (dict(scale=sc, policy=p),
         dict(N=300, scheme='secmac', secmac=dict(mode_select=False), policy=p, a2_rate=0.1,
              ntn_delay_scale=sc, **RUN))
        for sc in (1.0, 45.0) for p in ('static', 'cap', 'tvw')])
    # context poisoning: bogus-traffic injection
    E['poison'] = dict(reps=15, points=[
        (dict(N=N, A=A, scheme=s),
         dict(N=N, a4_load=A, **({'scheme': 'secmac'} if s in ('secmac', 'regulated') else {'scheme': s}),
              **({'secmac': dict(trust_aware=False)} if s == 'regulated' else {}), **RUN_SHORT))
        for N in (600, 1200) for A in (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
        for s in ('secmac', 'regulated', 'slotted', 'tdma')])
    E['poison_weak'] = dict(reps=15, points=[
        (dict(A=0.4, scheme=s),
         dict(N=600, a4_load=0.4, adv_gain_db=-6.0, scheme='secmac',
              **({'secmac': dict(trust_aware=False)} if s == 'regulated' else {}), **RUN_SHORT))
        for s in ('secmac', 'regulated')])
    # on-off injection: stability of the level and the mode
    E['pulsed'] = dict(reps=15, points=[
        (dict(variant=v), dict(N=600, a4_load=0.6, a4_on=60.0, a4_off=60.0, **variant(v), **RUN))
        for v in ('full', 'no_stability')])
    # relay adversary dragging the temporal profile
    drag_run = dict(T_warm=600.0, T_meas=3 * 3600.0, T_tail=300.0)
    E['drag'] = dict(reps=15, points=[
        (dict(variant='fast'), dict(N=300, scheme='secmac', secmac=dict(mode_select=False), a3_frac=0.1,
                                    a3_mode='slack', a3_start=600.0, **drag_run)),
        (dict(variant='fast_nocheck'), dict(N=300, scheme='secmac', secmac=dict(mode_select=False), a3_frac=0.1,
                                            a3_mode='slack', a3_start=600.0, casam=dict(zeta=1e9), **drag_run)),
        (dict(variant='patient'), dict(N=300, scheme='secmac', secmac=dict(mode_select=False), a3_frac=0.1,
                                       a3_mode='rate', a3_start=600.0, **drag_run)),
        (dict(variant='fast_amc'), dict(N=300, scheme='secmac', a3_frac=0.1, a3_mode='slack',
                                        a3_start=600.0, **drag_run)),
    ])
    # sensitivity
    E['sens_kappa'] = dict(reps=10, points=[
        (dict(kappa=k, attack=a),
         dict(N=300, scheme='secmac', secmac=dict(mode_select=False), casam=dict(kappa_S=(k, k, k)),
              **({'a2_rate': 0.1, 'replay_delay': ('fixed', 0.005)} if a else {}), **RUN))
        for k in (2.0, 3.0, 4.0, 6.0) for a in (0, 1)])
    E['sens_amc'] = dict(reps=10, points=[
        (dict(theta_c=tc, eta_m=em), dict(N=1200, scheme='secmac', amc=dict(theta_c=tc, eta_m=em), **RUN))
        for tc, em in ((0.3, 0.15), (0.2, 0.15), (0.5, 0.15), (0.3, 0.05), (0.3, 0.5))])
    # ablation without attack and under injection
    E['ablation'] = dict(reps=15, points=[
        (dict(variant=v, A=A), dict(N=1200, a4_load=A, **variant(v), **RUN_SHORT))
        for v in ('full', 'no_trust', 'no_stability', 'no_tvw', 'no_admission', 'slotted_casam', 'pure_casam')
        for A in (0.0, 0.4)])
    return E


def run_one(args):
    name, point, cfg, rep, seed = args
    cfg = dict(CFG, **cfg)
    if isinstance(cfg.get("replay_delay"), list):
        cfg["replay_delay"] = tuple(cfg["replay_delay"])
    cfg['seed'] = seed
    t0 = time.time()
    r = Simulator(cfg).run()
    out = dict(experiment=name, point=point, rep=rep, seed=seed, wall=time.time() - t0)
    keep = {k: v for k, v in r.items() if k not in ('trace', 'relay_log', 'a2_bins', 'sf_hist')}
    out.update(keep)
    if name == 'calibration':
        out['Psi'] = [x[6] for x in r['trace'] if x[0] >= cfg['T_warm']]
        out['T_c'] = [x[5] for x in r['trace'] if x[0] >= cfg['T_warm']]
    if name.startswith('replay_delay'):
        out['a2_bins'] = {str(k): v for k, v in r['a2_bins'].items()}
    if name == 'drag':
        out['relay_log'] = r['relay_log']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--scale', type=float, default=1.0, help='fraction of the replications')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    E = experiments()
    names = a.names or list(E)
    os.makedirs(RAW, exist_ok=True)
    for name in names:
        exp = E[name]
        reps = max(1, int(round(exp['reps'] * a.scale)))
        jobs = [(name, point, cfg, r, SEEDS[name] + r) for point, cfg in exp['points'] for r in range(reps)]
        t0 = time.time()
        path = os.path.join(RAW, name + '.jsonl')
        with Pool(a.workers) as pool, open(path, 'w') as f:
            for i, res in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
                f.write(json.dumps(res) + '\n')
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    print(f"{name}: {i + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
        print(f"{name}: done, {len(jobs)} runs in {time.time() - t0:.0f} s -> {path}", flush=True)


if __name__ == '__main__':
    main()
