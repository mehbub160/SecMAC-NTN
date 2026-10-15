"""CASAM: Context-Aware Security and Stability Control Module (network-server side).

Per decoded frame (Algorithm 1 of the paper):
  1. reconstruct counter c and stamp tau, verify the MIC            -> FORGED
  2. counter freshness: c > c_max (new) or c == c_max, j > j_acc (dup) -> REPLAY
  3. temporal validity window on the frame age (can only reject)      -> STALE
  4. dup: re-acknowledge without delivery                             -> REACK
  5. accept, update counters, authenticated accounting, time profile  -> ACCEPT
Per epoch and logical channel: load consistency Psi, context trust T_c and
security level S with hysteresis.

The same logic is used for all security policies; the policy only changes
which bound the window applies:
  'counter' : no window (LoRaWAN counter check only; frames carry no stamp)
  'static'  : one bound per link, sized for the worst case over a sync interval
  'cap'     : per-device bound from the time and accuracy of its last sync
  'tvw'     : SecMAC-NTN: min(cap, per-device temporal profile)
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .crypto import reconstruct_counter, reconstruct_stamp, TAU_MOD

FORGED, REPLAY, STALE, REACK, ACCEPT = 'forged', 'replay', 'stale', 'reack', 'accept'

DEFAULT_CASAM = dict(
    policy='tvw',
    rho_max=20e-6,        # crystal tolerance bound used by the NS
    eps_ts=0.100,         # DeviceTimeAns worst-case accuracy (LoRaWAN 1.0.4)
    eps_b=0.003,          # beacon timing accuracy of a slotted (beacon-tracking) device
    eps0=0.003,           # stamp quantisation (1 ms) plus 4 sigma of stamp jitter
    kappa_S=(4.0, 3.5, 3.0),   # profile margin per security level S = 0, 1, 2
    g_S=(0.5, 0.25, 0.0),      # profile gain per security level
    kappa_c=2.0,          # innovation clip
    kappa_0=4.0,          # fresher-than-predicted restart threshold
    g_sigma=0.2,
    sigma_0=0.003, sigma_min=0.001, sigma_max=0.010,
    zeta=0.5,             # drift anomaly: estimated drift > (1+zeta) rho_max flags the device
    T_w=1800.0,           # horizon of the drift consistency check
    r_max=3,
    # trust and level
    psi_G=1.5, psi_a=0.10, psi_d=0.05,   # psi_G: above the 99th percentile of attack-free Psi (calibration)
    w=(1/3, 1/3, 1/3), xi=0.2, g0=0.02,
    theta1=0.75, theta2=0.50, hyst=0.05, n_d=3,
    xi_d=0.9,             # discount of the authenticated accounting sums
    F_min=20.0,           # accepted frames (discounted) needed before Psi counts as evidence
    T_sync=6 * 3600.0,    # DeviceTime interval (for the static bound)
)


@dataclass
class DevState:
    c_max: int = -1
    j_acc: int = -1
    t_last: float = 0.0
    # clock synchronisation known to the NS
    t_sync: float = 0.0
    eps_sync: float = 0.100
    pending_sync_t: Optional[float] = None   # time of the ACK that carried a DeviceTimeAns
    # temporal profile
    prof: bool = False
    o_hat: float = 0.0
    sigma: float = 0.003
    t0: float = 0.0
    o0: float = 0.0
    flagged: bool = False
    beacon_synced: bool = False


@dataclass
class ChanEpoch:
    f: int = 0        # counter increments observed (frames transmitted by authenticated devices)
    a: int = 0        # attempts behind them
    n_dec: int = 0    # decoded frames
    n_forged: int = 0
    n_auth: int = 0   # frames with a valid MIC
    n_rej: int = 0    # authenticated frames rejected as replay or stale
    n_stale: int = 0
    n_replay: int = 0


@dataclass
class ChanState:
    T_air: float
    d_bar: float
    Delta: float
    d_max: float
    link: str
    F: float = 0.0
    A: float = 0.0
    X: float = 0.0
    T_c: float = 1.0
    S: int = 0
    down_ok: int = 0
    phi_G: float = 1.0
    Psi: float = 0.0
    G_auth: float = 0.0
    m_hat: float = 1.0
    B: float = 0.0        # discounted busy time (observed load, CAE)
    G_phy: float = 0.0
    ep: ChanEpoch = field(default_factory=ChanEpoch)
    level_changes: int = 0


class Casam:
    def __init__(self, cfg: dict, chans: Dict[int, dict], n_dev: int, dev_chan):
        self.cfg = dict(DEFAULT_CASAM)
        self.cfg.update(cfg or {})
        self.ch: Dict[int, ChanState] = {k: ChanState(**v) for k, v in chans.items()}
        self.dev = [DevState() for _ in range(n_dev)]
        self.dev_chan = dev_chan
        self.mode = {k: 'pure' for k in chans}     # access mode of the channel (from AMC)
        for d in self.dev:
            d.eps_sync = self.cfg['eps_ts']

    # ---- events from the rest of the network server -----------------------
    def on_devtime_ans(self, i: int, t: float):
        """The NS answered a DeviceTimeReq of device i in the ACK sent at t."""
        self.dev[i].pending_sync_t = t

    def on_beacon(self, k: int, t: float, dev_ids):
        """Devices of a slotted channel re-synchronise their clock on the beacon."""
        if self.mode[k] != 'slotted':
            return
        for i in dev_ids:
            d = self.dev[i]
            d.t_sync = t
            d.eps_sync = self.cfg['eps_b']
            d.beacon_synced = True
            d.prof = False
            d.flagged = False

    def set_mode(self, k: int, mode: str, dev_ids=()):
        if mode == 'pure' and self.mode[k] == 'slotted':
            # devices stop tracking the beacon; their cap keeps growing from the
            # last beacon and a profile may be re-established from accepted frames
            for i in dev_ids:
                self.dev[i].beacon_synced = False
                self.dev[i].prof = False
        self.mode[k] = mode

    # ---- temporal validity window ------------------------------------------
    def cap_bound(self, d: DevState, c: ChanState, t: float) -> float:
        # the delay uncertainty enters twice: in the synchronisation (the device
        # cannot measure its own path delay) and in the uplink of the frame
        return 2 * c.Delta + d.eps_sync + self.cfg['rho_max'] * max(0.0, t - d.t_sync) + self.cfg['eps0']

    def window(self, i: int, k: int, t: float) -> float:
        """Allowed residual age x = t_r - tau - T_air - d_bar for device i at time t."""
        cfg = self.cfg
        d, c = self.dev[i], self.ch[k]
        pol = cfg['policy']
        if pol == 'counter':
            return math.inf
        if pol == 'static':
            return 2 * c.Delta + cfg['eps0'] + cfg['eps_ts'] + cfg['rho_max'] * cfg['T_sync']
        b = self.cap_bound(d, c, t)
        if pol == 'cap':
            return b
        if d.prof and not d.beacon_synced:
            kappa = cfg['kappa_S'][c.S]
            pb = d.o_hat + cfg['rho_max'] * max(0.0, t - d.t_last) + kappa * d.sigma + cfg['eps0']
            b = min(b, pb)
        return b

    # ---- per-frame check ------------------------------------------------------
    def check(self, fr, t_r: float) -> str:
        cfg = self.cfg
        i, k = fr.dev, fr.k
        d, c = self.dev[i], self.ch[k]
        ep = c.ep
        ep.n_dec += 1
        # 1. reconstruction and MIC
        c32 = reconstruct_counter(fr.c16, max(d.c_max, 0))
        if cfg['policy'] == 'counter':
            tau_ok = True
        else:
            expected_ms = (t_r - c.T_air - c.d_bar - (d.o_hat if d.prof else 0.0)) * 1000.0
            tau32 = reconstruct_stamp(fr.tau20, expected_ms)
            tau_ok = (tau32 == fr.tau32)
        if not (fr.key_ok and c32 == fr.c and tau_ok):
            ep.n_forged += 1
            return FORGED
        ep.n_auth += 1
        # the device reports that it applied the DeviceTimeAns the NS sent at
        # pending_sync_t; the frame is judged against that correction, and the
        # state is committed only if the frame is accepted
        pending_reset = bool(fr.sync_flag) and d.pending_sync_t is not None and cfg['policy'] != 'counter'
        # 2. counter freshness
        new = c32 > d.c_max
        dup = (c32 == d.c_max) and (fr.j > d.j_acc)
        if not new and not dup:
            ep.n_rej += 1
            ep.n_replay += 1
            return REPLAY
        # 3. temporal validity window (can only reject; state unchanged on rejection)
        if cfg['policy'] != 'counter':
            x = t_r - fr.tau32 / 1000.0 - c.T_air - c.d_bar
            if pending_reset:
                bound = (2 * c.Delta + cfg['eps_ts'] + cfg['rho_max'] * max(0.0, t_r - d.pending_sync_t)
                         + cfg['eps0']) if cfg['policy'] != 'static' else self.window(i, k, t_r)
            else:
                bound = self.window(i, k, t_r)
            fr.cap_only = pending_reset or (not d.prof) or d.beacon_synced   # statistics only
            if x > bound:
                ep.n_rej += 1
                ep.n_stale += 1
                return STALE
        if pending_reset:
            d.t_sync = d.pending_sync_t
            d.eps_sync = cfg['eps_ts']
            d.pending_sync_t = None
            d.prof = False
            d.flagged = False
            d.beacon_synced = False
        # 4. duplicate of an accepted frame whose ACK was lost
        if dup:
            ep.a += fr.j - d.j_acc
            d.j_acc = fr.j
            return REACK
        # 5. accept
        gap = c32 - d.c_max            # frames transmitted since the last accepted one
        ep.f += gap
        ep.a += (fr.j + 1) + (gap - 1) * (cfg['r_max'] + 1)
        d.c_max, d.j_acc = c32, fr.j
        if cfg['policy'] == 'tvw' and not d.beacon_synced:
            self._update_profile(d, c, x, t_r)
        d.t_last = t_r
        return ACCEPT

    def _update_profile(self, d: DevState, c: ChanState, x: float, t: float):
        cfg = self.cfg
        if not d.prof:
            d.prof, d.o_hat, d.sigma = True, x, cfg['sigma_0']
            d.t0, d.o0, d.flagged = t, x, False
            return
        if x < d.o_hat - cfg['kappa_0'] * d.sigma:
            # fresher than predicted: restart the estimate (the window can only
            # tighten) but keep the drift checkpoint and the flag
            d.o_hat, d.sigma = x, cfg['sigma_0']
            return
        if d.flagged:
            return                                   # profile frozen
        g = cfg['g_S'][c.S]
        # the innovation may legitimately contain the drift since the last frame
        e = min(x - d.o_hat, cfg['kappa_c'] * d.sigma + cfg['rho_max'] * max(0.0, t - d.t_last))
        d.o_hat += g * e
        s2 = (1 - cfg['g_sigma']) * d.sigma ** 2 + cfg['g_sigma'] * e * e
        d.sigma = min(max(math.sqrt(s2), cfg['sigma_min']), cfg['sigma_max'])
        # drift consistency over a sliding horizon T_w: a profile that moves faster
        # than a crystal can drift is frozen (the device is flagged)
        dt = t - d.t0
        if dt >= cfg['T_w']:
            if (d.o_hat - d.o0) / dt > (1 + cfg['zeta']) * cfg['rho_max']:
                d.flagged = True
            d.t0, d.o0 = t, d.o_hat

    def slack(self, i: int, t: float) -> float:
        """Room between the current window and the device's expected residual (oracle for the relay adversary)."""
        k = self.dev_chan[i]
        return self.window(i, k, t)

    # ---- per-epoch trust, level and authenticated context ---------------------
    def epoch(self, k: int, busy: float, T_e: float, t: float):
        """CAE context update and CASAM trust/level update for channel k.
        busy: seconds in which the channel was occupied during the epoch."""
        cfg = self.cfg
        c = self.ch[k]
        ep = c.ep
        xd = cfg['xi_d']
        # observed load: discounted busy fraction inverted for Poisson overlap
        c.B = xd * c.B + busy
        c.F = xd * c.F + ep.f
        c.A = xd * c.A + ep.a
        c.X = xd * c.X + T_e
        b = min(max(c.B / c.X, 0.0), 0.999) if c.X > 0 else 0.0
        G_phy = -math.log(1.0 - b)
        c.G_phy = G_phy
        c.m_hat = (c.A / c.F) if c.F > 0 else 1.0
        c.G_auth = c.A * c.T_air / c.X if c.X > 0 else 0.0
        # load consistency is evidence only once enough authenticated frames back it
        reliable = c.F >= cfg['F_min']
        c.Psi = abs(G_phy - c.G_auth) / max(c.G_auth, cfg['g0']) if reliable else 0.0
        z_a = ep.n_forged / ep.n_dec if ep.n_dec > 0 else 0.0
        z_d = ep.n_rej / ep.n_auth if ep.n_auth > 0 else 0.0
        phi_G = math.exp(-c.Psi / cfg['psi_G'])
        phi_a = math.exp(-z_a / cfg['psi_a'])
        phi_d = math.exp(-z_d / cfg['psi_d'])
        c.phi_G = phi_G
        w = cfg['w']
        c.T_c = cfg['xi'] * c.T_c + (1 - cfg['xi']) * (w[0] * phi_G + w[1] * phi_a + w[2] * phi_d)
        # level with immediate escalation and hysteresis on the way down
        target = 2 if c.T_c < cfg['theta2'] else (1 if c.T_c < cfg['theta1'] else 0)
        S_old = c.S
        if target > c.S:
            c.S, c.down_ok = target, 0
        elif target < c.S:
            thr = cfg['theta2'] if c.S == 2 else cfg['theta1']
            if c.T_c > thr + cfg['hyst']:
                c.down_ok += 1
                if c.down_ok >= cfg['n_d']:
                    c.S, c.down_ok = c.S - 1, 0
            else:
                c.down_ok = 0
        else:
            c.down_ok = 0
        if c.S != S_old:
            c.level_changes += 1
        # trust of the observed load handed to the AMC: direct forgery evidence only
        # (Psi is biased when weak devices are rarely heard, so it is kept for T_c)
        c.phi_G = phi_a
        stats = dict(G_phy=G_phy, G_auth=c.G_auth, m_hat=c.m_hat, Psi=c.Psi, T_c=c.T_c, S=c.S,
                     phi_G=c.phi_G, reliable=reliable,
                     z_a=z_a, z_d=z_d, n_dec=ep.n_dec, n_auth=ep.n_auth, n_forged=ep.n_forged,
                     n_stale=ep.n_stale, n_replay=ep.n_replay)
        c.ep = ChanEpoch()
        return stats
