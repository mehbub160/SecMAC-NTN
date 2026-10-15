"""Validate the collision model of the simulator against ALOHA theory.

With capture, retransmissions and the duty cycle disabled, a single logical
channel with Poisson arrivals must reproduce S = G e^{-2G} (pure) and
S = G e^{-G} (slotted, with the slot guard removed).  Run:  python3 validate_aloha.py
"""
import math
import sys

from secmac.sim import Simulator


def run(scheme, N, demand, seed):
    cfg = dict(N=N, frac_ntn=0.0, R_ter=200.0, shadow=0.0,     # all devices SF7 terrestrial
               demand=demand, queue=1, duty=1.0, r_max=0, ack_loss=0.0,
               capture_db=200.0, scheme=scheme, policy='counter', stamp=False,
               sigma_s=0.0, T_warm=300, T_meas=3600, T_tail=60, seed=seed,
               a1_rate=0, a2_rate=0, a4_load=0)
    s = Simulator(cfg)
    r = s.run()
    # offered load G: attempts per airtime, measured
    T_air = s.chans[0].T_air
    G = r['attempts'] * T_air / cfg['T_meas']
    S = r['accepted'] * T_air / cfg['T_meas']
    return G, S


if __name__ == '__main__':
    print(f"{'scheme':8s} {'N':>5s} {'G':>7s} {'S_sim':>7s} {'S_theory':>9s} {'rel.err':>8s}")
    worst = 0.0
    for scheme in ('pure', 'slotted'):
        for N, demand in ((100, 0.002), (100, 0.005), (200, 0.005), (300, 0.005)):
            Gs, Ss = [], []
            for seed in range(3):
                G, S = run(scheme, N, demand, seed)
                Gs.append(G)
                Ss.append(S)
            G, S = sum(Gs) / 3, sum(Ss) / 3
            th = G * math.exp(-2 * G) if scheme == 'pure' else G * math.exp(-G)
            err = abs(S - th) / th
            worst = max(worst, err)
            print(f"{scheme:8s} {N:5d} {G:7.3f} {S:7.4f} {th:9.4f} {100*err:7.1f}%")
    print(f"worst relative error {100*worst:.1f}%")
    sys.exit(0 if worst < 0.05 else 1)
