"""Time the per-frame work of the network server with real AES-CMAC.

    python3 bench_verify.py   -> results/bench.json (microseconds per verified frame)

The simulator checks MIC semantics without running AES; this script runs the
real path (counter and stamp reconstruction, AES-128-CMAC over the frame, the
checks of Algorithm 1) on one core to give the constant per-frame cost.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secmac import crypto  # noqa: E402
from secmac.casam import Casam, ACCEPT  # noqa: E402
from secmac.phy import airtime  # noqa: E402

KEY = bytes(range(16))
DEV = 0x26011BDA


class Fr:
    pass


def main(n=20000):
    T_air = airtime(7, 54)
    cas = Casam(dict(policy='tvw'), {0: dict(T_air=T_air, d_bar=0.0, Delta=0.0, d_max=1e-5, link='ter')}, 1, [0])
    cas.dev[0].t_sync = 0.0
    msg = bytes(50)
    frames = []
    t = 1000.0
    for c in range(n):
        tau = int((t - T_air) * 1000)
        mic = crypto.compute_mic(KEY, DEV, c, tau, msg)
        frames.append((c & 0xFFFF, tau & 0xFFFFF, mic, t))
        t += 20.0
    t0 = time.perf_counter()
    acc = 0
    for c16, tau20, mic, tr in frames:
        d = cas.dev[0]
        c = crypto.reconstruct_counter(c16, max(d.c_max, 0))
        tau = crypto.reconstruct_stamp(tau20, (tr - T_air - (d.o_hat if d.prof else 0.0)) * 1000.0)
        ok = crypto.compute_mic(KEY, DEV, c, tau, msg) == mic
        fr = Fr()
        fr.dev, fr.k, fr.c, fr.c16, fr.tau32, fr.tau20, fr.j = 0, 0, c, c16, tau, tau20, 0
        fr.sync_flag, fr.key_ok, fr.kind = 0, ok, 'data'
        if cas.check(fr, tr) == ACCEPT:
            acc += 1
    dt = time.perf_counter() - t0
    us = 1e6 * dt / n
    print(f'{n} frames verified, {acc} accepted, {us:.1f} us per frame (AES-CMAC included)')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'bench.json')
    json.dump(dict(frames=n, accepted=acc, us_per_frame=us, python=sys.version.split()[0]), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
