"""Packet-level discrete-event simulator for SecMAC-NTN and its baselines.

One run simulates N end devices, half in a terrestrial cell and half in the
footprint of one LEO satellite, that send confirmed uplinks to one network
server (NS).  Everything below is per *logical channel* k = (link type, SF).

Modelled: LoRa airtime, SF from the link budget, co-SF collisions with a 6 dB
capture threshold against the summed interference, Poisson traffic with a
device queue, 1 % duty cycle, confirmed uplinks with binary exponential
backoff and a retry limit, per-device propagation delay, device clocks with
offset, drift and stamp jitter, DeviceTime and beacon synchronisation, five
access schemes (pure ALOHA, slotted ALOHA, reservation, ideal TDMA, SecMAC-NTN)
and four adversaries (A1 duplicate replay, A2 suppress-and-replay, A3 relay,
A4 bogus-traffic injection).

Not modelled (see the paper): orbital motion and visibility, Doppler, fading,
imperfect SF orthogonality, downlink contention, loss of the AMC broadcast.
"""
import heapq
import math
import random
from collections import deque

from . import phy
from .casam import Casam, ACCEPT, REACK, STALE, REPLAY, FORGED
from .amc import Amc

# event kinds
GEN, TRY, TX, RXS, RXE, ACK, TOUT, EPOCH, BEACON, ADV, ROUND, REQEND = range(12)

DEFAULT = dict(
    # population and run length
    N=600, frac_ntn=0.5, T_warm=600.0, T_meas=3600.0, T_tail=600.0, seed=1,
    # topology
    R_ter=3000.0, h_sat=600e3, elev_min=30.0, feeder=800e3, G_sat=14.0, L_extra=6.0,
    margin=3.0, shadow=7.8, f_c=868e6, ptx=phy.TX_POWER_DBM, ntn_delay_scale=1.0,
    # traffic
    demand=0.005, queue=4, duty=0.01,
    # frames
    bytes_lorawan=50, stamp_bytes=4, ack_bytes=13, req_bytes=8, beacon_bytes=17,
    # link
    capture_db=6.0, ack_loss=0.05, rx_delay=1.0, rx_listen=0.03,
    # MAC
    r_max=3, W0=8, Wd=8, beacon_period=128.0, sigma_s=0.002,
    # clocks
    rho_max=20e-6, eps_ts=0.100, eps_b=0.003, T_sync=6 * 3600.0, jitter=0.0005,
    # scheme: pure | slotted | reservation | tdma | secmac
    scheme='secmac',
    secmac=dict(trust_aware=True, stability=True, admission=True, mode_select=True),
    policy='tvw', stamp=True,
    casam={}, amc={}, T_e=60.0,
    # reservation (RESS-IoT type)
    TW=100, n_grant=3, sift_mean=0.12,
    # adversaries
    a1_rate=0.0, a2_rate=0.0, replay_delay=('loguniform', 1e-3, 100.0),
    a4_load=0.0, a4_on=0.0, a4_off=0.0, a3_frac=0.0, a3_mode='slack', a3_factor=0.9,
    a3_start=0.0, adv_gain_db=3.0,
    trace=False,
)


class Frame:
    __slots__ = ('dev', 'c', 'c16', 'tau32', 'tau20', 'j', 'sync_flag', 'sync_req', 'key_ok', 'k',
                 'T_air', 'prx', 't_tx', 'rs', 're', 'kind', 'gen_t', 'suppressed', 'd', 'token',
                 'added', 'slot', 'cap_only')

    def __init__(self, dev, k, T_air, prx, kind, gen_t=0.0):
        self.dev, self.k, self.T_air, self.prx, self.kind, self.gen_t = dev, k, T_air, prx, kind, gen_t
        self.c = self.c16 = self.tau32 = self.tau20 = self.j = 0
        self.sync_flag = self.sync_req = 0
        self.key_ok = True
        self.t_tx = self.rs = self.re = 0.0
        self.suppressed = False
        self.d = 0.0
        self.token = 0
        self.added = 0.0
        self.slot = 0
        self.cap_only = False


class Device:
    __slots__ = ('id', 'link', 'sf', 'k', 'delay', 'prx', 'T_air', 'lam', 'queue', 'fcnt', 'cur',
                 'state', 't_dc', 'o_sync', 'rho', 't_sync', 'sync_due', 'sync_pending',
                 'slot_err', 't_beacon', 'tracking', 'energy', 'token', 'idx', 'victim', 'x', 'y',
                 'eps_cur')

    def __init__(self, i):
        self.id = i
        self.queue = deque()
        self.fcnt = 0
        self.cur = None
        self.state = 0        # 0 idle, 1 waiting (scheduled tx or ack), 2 backoff
        self.t_dc = 0.0
        self.sync_due = False
        self.sync_pending = False
        self.slot_err = 0.0
        self.t_beacon = -1e9
        self.tracking = False
        self.energy = 0.0
        self.token = 0
        self.victim = False


class Channel:
    __slots__ = ('k', 'link', 'sf', 'T_air', 'T_req', 'T_ack', 'T_beacon', 'devs', 'active', 'busy',
                 'busy_until', 'mode', 'p', 'T_slot', 'd_bar', 'Delta', 'd_max', 'd_min',
                 'round_t', 'contenders', 'requests', 'prx_med', 'period', 'bogus_next', 'a4_on')

    def __init__(self, k, link, sf):
        self.k, self.link, self.sf = k, link, sf
        self.devs = []
        self.active = []
        self.busy = 0.0
        self.busy_until = -1.0
        self.mode, self.p = 'pure', 1.0
        self.contenders, self.requests = [], []
        self.round_t = 0.0
        self.a4_on = True


def _dbm_to_mw(x):
    return 10 ** (x / 10.0)


class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = c = dict(DEFAULT)
        c.update(cfg or {})
        c['secmac'] = dict(DEFAULT['secmac'], **(cfg.get('secmac', {}) if cfg else {}))
        self.rng = random.Random(c['seed'])
        self.arng = random.Random(c['seed'] + 7919)      # adversary stream
        self.t = 0.0
        self.heap = []
        self.seq = 0
        self.T_end = c['T_warm'] + c['T_meas'] + c['T_tail']
        self.frame_bytes = c['bytes_lorawan'] + (c['stamp_bytes'] if (c['stamp'] and c['policy'] != 'counter') else 0)
        self._build_topology()
        self._build_ns()
        self._init_stats()

    # ------------------------------------------------------------------ setup
    def _build_topology(self):
        c, rng = self.cfg, self.rng
        N = c['N']
        n_ntn = int(round(N * c['frac_ntn']))
        self.devs = []
        self.chans = {}
        d_lo, d_hi = phy.ntn_delay_bounds_s(c['h_sat'], c['elev_min'], c['feeder'])
        d_lo *= c['ntn_delay_scale']
        d_hi *= c['ntn_delay_scale']
        R_g = phy.ground_radius_for_elevation_m(c['elev_min'], c['h_sat'])
        for i in range(N):
            dv = Device(i)
            if i < N - n_ntn:
                dv.link = 'ter'
                while True:
                    d = max(50.0, c['R_ter'] * math.sqrt(rng.random()))
                    prx = c['ptx'] - phy.terrestrial_pathloss_db(d, rng.gauss(0, c['shadow']))
                    sf = phy.choose_sf(prx, c['margin'])
                    if sf is not None:
                        break
                dv.delay = d / phy.C_LIGHT
            else:
                dv.link = 'ntn'
                while True:
                    d = R_g * math.sqrt(rng.random())
                    sr = phy.slant_range_m(d, c['h_sat'])
                    prx = c['ptx'] + c['G_sat'] - phy.fspl_db(sr, c['f_c']) - c['L_extra']
                    sf = phy.choose_sf(prx, c['margin'])
                    if sf is not None:
                        break
                dv.delay = ((sr + c['feeder']) / phy.C_LIGHT) * c['ntn_delay_scale']
            dv.sf, dv.prx = sf, prx
            dv.k = (0 if dv.link == 'ter' else 1) * 6 + (sf - 7)
            dv.T_air = phy.airtime(sf, self.frame_bytes)
            dv.lam = c['demand'] / dv.T_air
            # clock: random phase of the DeviceTime interval, offset within the sync accuracy, drift
            dv.t_sync = -rng.random() * c['T_sync']
            dv.o_sync = rng.uniform(-c['eps_ts'], c['eps_ts'])
            dv.eps_cur = c['eps_ts']
            dv.rho = rng.uniform(-c['rho_max'], c['rho_max'])
            ch = self.chans.get(dv.k)
            if ch is None:
                ch = self.chans[dv.k] = Channel(dv.k, dv.link, sf)
                ch.T_air = dv.T_air
                ch.T_req = phy.airtime(sf, c['req_bytes'])
                ch.T_ack = phy.airtime(sf, c['ack_bytes'])
                ch.T_beacon = phy.airtime(9, c['beacon_bytes'])
                if dv.link == 'ter':
                    ch.d_min, ch.d_max = 0.0, c['R_ter'] / phy.C_LIGHT
                else:
                    ch.d_min, ch.d_max = d_lo, d_hi
                ch.d_bar = 0.5 * (ch.d_min + ch.d_max)
                ch.Delta = 0.5 * (ch.d_max - ch.d_min)
            dv.idx = len(ch.devs)
            ch.devs.append(i)
            self.devs.append(dv)
        self.d_max_link = {'ter': c['R_ter'] / phy.C_LIGHT, 'ntn': d_hi}
        for ch in self.chans.values():
            prxs = sorted(self.devs[i].prx for i in ch.devs)
            ch.prx_med = prxs[len(prxs) // 2] + c['adv_gain_db']
            ch.T_slot = ch.T_air + 6 * c['sigma_s'] + 2 * ch.Delta
            ch.period = len(ch.devs) * ch.T_slot            # ideal TDMA frame
        # relay victims
        if c['a3_frac'] > 0:
            for dv in self.devs:
                dv.victim = self.arng.random() < c['a3_frac']

    def _build_ns(self):
        c = self.cfg
        cas_cfg = dict(c['casam'])
        cas_cfg.update(policy=c['policy'], rho_max=c['rho_max'], eps_ts=c['eps_ts'], eps_b=c['eps_b'],
                       r_max=c['r_max'], T_sync=c['T_sync'])
        chans = {k: dict(T_air=ch.T_air, d_bar=ch.d_bar, Delta=ch.Delta, d_max=ch.d_max, link=ch.link)
                 for k, ch in self.chans.items()}
        self.cas = Casam(cas_cfg, chans, len(self.devs), [dv.k for dv in self.devs])
        for dv in self.devs:          # the NS knows when it last answered a DeviceTimeReq
            self.cas.dev[dv.id].t_sync = dv.t_sync
        self.amc = None
        sch = c['scheme']
        if sch == 'secmac':
            amc_cfg = dict(c['amc'])
            amc_cfg.update(c['secmac'])
            amc_cfg.update(sigma_s=c['sigma_s'], duty=c['duty'])
            self.amc = Amc(amc_cfg, {k: dict(N=len(ch.devs), T_air=ch.T_air, Delta=ch.Delta)
                                     for k, ch in self.chans.items()})
            for k, ch in self.chans.items():
                ch.mode, ch.p = self.amc.ch[k].M, self.amc.ch[k].p
                self.cas.set_mode(k, ch.mode)
        elif sch == 'slotted':
            for k, ch in self.chans.items():
                ch.mode = 'slotted'
                self.cas.set_mode(k, 'slotted')
        else:
            for ch in self.chans.values():
                ch.mode = 'pure'

    def _init_stats(self):
        self.st = dict(generated=0, accepted=0, dropped_queue=0, dropped_retry=0, latency=0.0,
                       attempts=0, collided=0, energy=0.0, false_stale=0, legit_checked=0,
                       a1_att=0, a1_acc=0, a2_att=0, a2_acc=0, a2_acc_caponly=0, bogus_dec=0, bogus_acc=0,
                       relay_att=0, relay_acc=0, relay_frames=0, relay_window_open=0,
                       dup_att=0, dup_acc=0, dup_reack=0,
                       epochs=0, level_epochs=[0, 0, 0], attacked_level_epochs=[0, 0, 0],
                       attacked_epochs=0, p_sum=0.0, p_n=0)
        self.a2_bins = {}        # replay delay bin -> [attempted, accepted]
        self.relay_log = []      # (t, accepted delay, sf, open)
        self.trace = []

    # ------------------------------------------------------------------ helpers
    def push(self, t, kind, a=None, b=None):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, a, b))

    def in_window(self, t):
        return self.cfg['T_warm'] <= t < self.cfg['T_warm'] + self.cfg['T_meas']

    def dev_time_ms(self, dv, t):
        o = dv.o_sync + dv.rho * (t - dv.t_sync)
        return int(math.floor((t - o + self.rng.gauss(0, self.cfg['jitter'])) * 1000.0))

    def energy(self, dv, joules, t):
        if self.in_window(t):
            self.st['energy'] += joules

    def replay_delay(self):
        spec = self.cfg['replay_delay']
        if spec[0] == 'loguniform':
            return math.exp(self.arng.uniform(math.log(spec[1]), math.log(spec[2])))
        return float(spec[1])

    # ------------------------------------------------------------------ run
    def run(self):
        c = self.cfg
        for dv in self.devs:
            self.push(self.rng.expovariate(dv.lam), GEN, dv.id)
        self.push(c['T_e'], EPOCH)
        self.push(0.0, BEACON)
        if c['scheme'] == 'reservation':
            for k in self.chans:
                self.push(self.rng.random() * 2.0, ROUND, k)
        if c['a4_load'] > 0:
            for k, ch in self.chans.items():
                self.push(self.arng.expovariate(c['a4_load'] / ch.T_air), ADV, ('bogus', k))
            if c['a4_on'] > 0:
                self.push(c['a4_on'], ADV, ('a4toggle', False))
        heap, T_end = self.heap, self.T_end
        while heap:
            t, _, kind, a, b = heapq.heappop(heap)
            if t > T_end:
                break
            self.t = t
            if kind == GEN:
                self.ev_gen(a, t)
            elif kind == TRY:
                self.ev_try(a, b, t)
            elif kind == TX:
                self.ev_tx(a, t)
            elif kind == RXS:
                self.ev_rxs(a, t)
            elif kind == RXE:
                self.ev_rxe(a, t)
            elif kind == ACK:
                self.ev_ack(a, b, t)
            elif kind == TOUT:
                self.ev_tout(a, b, t)
            elif kind == EPOCH:
                self.ev_epoch(t)
            elif kind == BEACON:
                self.ev_beacon(t)
            elif kind == ADV:
                self.ev_adv(a, t)
            elif kind == ROUND:
                self.ev_round(a, t)
            elif kind == REQEND:
                self.ev_reqend(a, t)
        return self.results()

    # ------------------------------------------------------------------ device events
    def ev_gen(self, i, t):
        c, dv = self.cfg, self.devs[i]
        self.push(t + self.rng.expovariate(dv.lam), GEN, i)
        if self.in_window(t):
            self.st['generated'] += 1
        if len(dv.queue) >= c['queue']:
            if self.in_window(t):
                self.st['dropped_queue'] += 1
            return
        dv.queue.append(t)
        if dv.state == 0 and dv.cur is None:
            self.schedule_try(dv, max(t, dv.t_dc))

    def schedule_try(self, dv, t):
        dv.token += 1
        dv.state = 1
        self.push(t, TRY, dv.id, dv.token)

    def ev_try(self, i, token, t):
        c, dv = self.cfg, self.devs[i]
        if token != dv.token:
            return
        if not dv.queue:
            dv.state = 0
            return
        ch = self.chans[dv.k]
        if dv.cur is None:
            fr = Frame(i, dv.k, dv.T_air, dv.prx, 'data', dv.queue[0])
            fr.c = dv.fcnt
            dv.fcnt += 1
            fr.j = 0
            dv.cur = fr
        fr = dv.cur
        if t < dv.t_dc:
            self.schedule_try(dv, dv.t_dc)
            return
        sch = c['scheme']
        if sch == 'tdma':
            n = math.floor((t - dv.idx * ch.T_slot) / ch.period)
            t_slot = n * ch.period + dv.idx * ch.T_slot
            while t_slot < t:
                t_slot += ch.period
            self.push(t_slot, TX, i)
            return
        if sch == 'reservation':
            dv.state = 0
            return                      # the round machinery picks the device up
        mode = ch.mode
        p = ch.p if sch == 'secmac' else 1.0
        if p < 1.0 and self.rng.random() >= p:
            self.schedule_try(dv, t + (0.5 + self.rng.random()) * dv.T_air / c['duty'])
            return
        if mode == 'slotted':
            if not dv.tracking or t - dv.t_beacon > 2 * c['beacon_period']:
                dv.tracking = True        # start tracking at the next beacon, then desynchronise
                nb = (math.floor(t / c['beacon_period']) + 1) * c['beacon_period']
                self.schedule_try(dv, nb + 1e-3 + self.rng.random() * dv.T_air / c['duty'])
                return
            n = math.floor((t - dv.slot_err) / ch.T_slot) + 1
            self.push(n * ch.T_slot + dv.slot_err, TX, i)
            return
        self.push(t, TX, i)

    def ev_tx(self, i, t):
        c, dv = self.cfg, self.devs[i]
        fr = dv.cur
        if fr is None:
            return
        ch = self.chans[dv.k]
        fr.t_tx = t
        fr.tau32 = self.dev_time_ms(dv, t) & 0xFFFFFFFF
        fr.tau20 = fr.tau32 & 0xFFFFF
        fr.c16 = fr.c & 0xFFFF
        if not dv.sync_due and t - dv.t_sync >= c['T_sync']:
            dv.sync_due = True
        fr.sync_req = 1 if dv.sync_due else 0
        fr.sync_flag = 1 if dv.sync_pending else 0
        fr.rs = t + dv.delay
        fr.re = fr.rs + fr.T_air
        fr.suppressed = False
        dv.token += 1
        fr.token = dv.token
        dv.t_dc = t + fr.T_air / c['duty']
        self.energy(dv, phy.P_TX_W * fr.T_air, t)
        if self.in_window(t):
            self.st['attempts'] += 1
        # adversaries acting on the air interface
        if dv.victim:
            self.relay_frame(dv, fr, ch, t)
        elif c['a2_rate'] > 0 and self.arng.random() < c['a2_rate']:
            self.suppress_and_replay(dv, fr, ch, t)
        if not fr.suppressed:          # a suppressed frame is not on the air at the receiver
            self.push(fr.rs, RXS, fr)
            self.push(fr.re, RXE, fr)
        # acknowledgement timeout (worst-case round trip of the link)
        t_out = fr.re + c['rx_delay'] + 2 * self.d_max_link[dv.link] + ch.T_ack + 0.05
        dv.state = 1
        self.push(t_out, TOUT, i, fr.token)

    def ev_ack(self, i, token, t):
        c, dv = self.cfg, self.devs[i]
        if token != dv.token or dv.cur is None:
            return
        ch = self.chans[dv.k]
        fr = dv.cur
        self.energy(dv, phy.P_RX_W * (c['rx_listen'] + ch.T_ack), t)
        if fr.sync_req:
            # DeviceTimeAns applied: new offset within the DeviceTime accuracy
            dv.o_sync = self.rng.uniform(-c['eps_ts'], c['eps_ts'])
            dv.t_sync = t
            dv.sync_due = False
            dv.sync_pending = True
            dv.eps_cur = c['eps_ts']
        elif fr.sync_flag:
            dv.sync_pending = False
        dv.queue.popleft()
        dv.cur = None
        dv.state = 0
        if dv.queue:
            self.schedule_try(dv, max(t, dv.t_dc))

    def ev_tout(self, i, token, t):
        c, dv = self.cfg, self.devs[i]
        if token != dv.token or dv.cur is None:
            return
        fr = dv.cur
        self.energy(dv, phy.P_RX_W * 2 * c['rx_listen'], t)
        fr.j += 1
        if fr.j > c['r_max']:
            if self.in_window(fr.gen_t):
                self.st['dropped_retry'] += 1
            dv.queue.popleft()
            dv.cur = None
            dv.state = 0
            if dv.queue:
                self.schedule_try(dv, max(t, dv.t_dc))
            return
        if c['scheme'] in ('reservation', 'tdma'):
            self.schedule_try(dv, max(t, dv.t_dc))
            return
        bo = self.rng.random() * c['W0'] * (2 ** fr.j) * fr.T_air
        self.schedule_try(dv, max(dv.t_dc, t) + bo)

    # ------------------------------------------------------------------ channel events
    def ev_rxs(self, fr, t):
        ch = self.chans[fr.k]
        if fr.rs >= ch.busy_until:
            ch.busy += fr.re - fr.rs
            ch.busy_until = fr.re
        elif fr.re > ch.busy_until:
            ch.busy += fr.re - ch.busy_until
            ch.busy_until = fr.re
        ch.active.append(fr)

    def ev_rxe(self, fr, t):
        c, ch = self.cfg, self.chans[fr.k]
        # interference from every other frame of the channel that overlaps in time
        p_self = _dbm_to_mw(fr.prx)
        interf = 0.0
        keep = []
        horizon = t - 3 * ch.T_air
        for g in ch.active:
            if g.re < horizon:
                continue
            keep.append(g)
            if g is fr:
                continue
            if g.rs < fr.re and g.re > fr.rs:
                interf += _dbm_to_mw(g.prx)
        ch.active = keep
        captured = (not fr.suppressed) and (interf == 0.0 or p_self >= interf * 10 ** (c['capture_db'] / 10.0))
        kind = fr.kind
        if kind == 'jam':
            return
        if not captured:
            if kind == 'data' and not fr.suppressed and self.in_window(t):
                self.st['collided'] += 1
            if kind == 'req':
                pass
            return
        if kind == 'req':
            ch.requests.append(fr)
            return
        if kind == 'bogus':
            self.st['bogus_dec'] += 1 if self.in_window(t) else 0
            out = self.cas.check(fr, t)
            if out in (ACCEPT, REACK):
                self.st['bogus_acc'] += 1
            return
        out = self.cas.check(fr, t)
        inw = self.in_window(t)
        if kind == 'data':
            if inw:
                self.st['legit_checked'] += 1
                if out == STALE:
                    self.st['false_stale'] += 1
            if out in (ACCEPT, REACK):
                self.deliver(fr, out, t)
            return
        # adversary-originated copies: a1 (duplicate), a2 (suppress-and-replay), relay
        acc = out in (ACCEPT, REACK)
        if kind == 'a1':
            if inw:
                self.st['a1_att'] += 1
                self.st['a1_acc'] += acc
        elif kind == 'a2':
            if inw:
                self.st['a2_att'] += 1
                self.st['a2_acc'] += acc
                self.st['a2_acc_caponly'] += acc and fr.cap_only
                b = self.a2_bins.setdefault(self._bin(fr.d), [0, 0])
                b[0] += 1
                b[1] += acc
        elif kind == 'relay':
            if inw:
                self.st['relay_att'] += 1
                self.st['relay_acc'] += acc
            if acc:
                dv = self.devs[fr.dev]
                self.relay_log.append((t, fr.added, dv.sf, 1 if fr.added >= fr.T_air else 0,
                                       1 if self.chans[dv.k].mode == 'slotted' else 0))
                self.deliver(fr, out, t)
        if acc and kind != 'relay':
            # the NS treats an accepted replay as legitimate: stale data is delivered
            # and the device is acknowledged (not counted as a delivered frame)
            self.deliver(fr, out, t)

    def _bin(self, d):
        return round(math.log10(d) * 4) / 4.0

    def deliver(self, fr, out, t):
        c = self.cfg
        dv = self.devs[fr.dev]
        if out == ACCEPT and fr.kind in ('data', 'relay') and self.in_window(fr.gen_t):
            self.st['accepted'] += 1
            self.st['latency'] += t - fr.gen_t
        # acknowledgement (downlink), possibly carrying a DeviceTimeAns
        if self.rng.random() >= c['ack_loss']:
            if fr.sync_req:
                self.cas.on_devtime_ans(fr.dev, t + c['rx_delay'] + dv.delay)
            self.push(t + c['rx_delay'] + dv.delay + self.chans[fr.k].T_ack, ACK, fr.dev, fr.token)
        if c['a1_rate'] > 0 and fr.kind == 'data' and out == ACCEPT and self.arng.random() < c['a1_rate']:
            self.schedule_copy(fr, 'a1', t + self.replay_delay())

    # ------------------------------------------------------------------ adversaries
    def adv_frame(self, fr, kind, t_tx, ch):
        g = Frame(fr.dev, fr.k, fr.T_air, ch.prx_med, kind, fr.gen_t)
        g.c, g.c16, g.tau32, g.tau20, g.j = fr.c, fr.c16, fr.tau32, fr.tau20, fr.j
        g.sync_flag, g.sync_req, g.key_ok, g.token = fr.sync_flag, 0, fr.key_ok, fr.token
        g.t_tx = t_tx
        g.rs = t_tx + ch.d_bar
        g.re = g.rs + g.T_air
        return g

    def schedule_copy(self, fr, kind, t_tx):
        ch = self.chans[fr.k]
        g = self.adv_frame(fr, kind, t_tx, ch)
        g.d = t_tx - (fr.t_tx + fr.T_air)
        self.push(g.rs, RXS, g)
        self.push(g.re, RXE, g)

    def suppress_and_replay(self, dv, fr, ch, t):
        """A2: a jamming burst prevents the reception of the frame; the recording
        is replayed d seconds after the end of the original."""
        fr.suppressed = True
        jam = Frame(-1, fr.k, fr.T_air, ch.prx_med, 'jam')
        jam.rs, jam.re = fr.rs, fr.re
        self.push(jam.rs, RXS, jam)
        self.push(jam.re, RXE, jam)
        self.schedule_copy(fr, 'a2', t + fr.T_air + self.replay_delay())

    def relay_frame(self, dv, fr, ch, t):
        """A3: the victim's direct path is absent; a relay forwards the frame with
        an added delay chosen to stay inside the current acceptance window."""
        c = self.cfg
        fr.suppressed = True
        self.st['relay_frames'] += 1
        o = dv.o_sync + dv.rho * (t - dv.t_sync)
        x_legit = o + (dv.delay - ch.d_bar)
        t_arr = t + dv.delay + fr.T_air
        w = self.cas.window(dv.id, dv.k, t_arr)
        slack = max(0.0, w - x_legit)
        if t < c['a3_start'] or w == math.inf:
            d = 0.0
        elif c['a3_mode'] == 'slack':
            d = c['a3_factor'] * slack
        else:   # 'rate': drag no faster than a crystal could drift, bounded by the slack
            d = min(c['a3_factor'] * c['rho_max'] * (t - c['a3_start']), 0.95 * slack)
        g = self.adv_frame(fr, 'relay', t + d, ch)
        g.rs = t + d + dv.delay
        g.re = g.rs + g.T_air
        g.added = d
        self.push(g.rs, RXS, g)
        self.push(g.re, RXE, g)

    def ev_adv(self, a, t):
        c = self.cfg
        what = a[0]
        if what == 'bogus':
            k = a[1]
            ch = self.chans[k]
            self.push(t + self.arng.expovariate(c['a4_load'] / ch.T_air), ADV, ('bogus', k))
            if not ch.a4_on:
                return
            g = Frame(ch.devs[self.arng.randrange(len(ch.devs))], k, ch.T_air, ch.prx_med, 'bogus')
            g.key_ok = False
            g.c16, g.tau20 = self.arng.randrange(1 << 16), self.arng.randrange(1 << 20)
            g.rs = t + ch.d_bar
            g.re = g.rs + g.T_air
            self.push(g.rs, RXS, g)
            self.push(g.re, RXE, g)
        elif what == 'a4toggle':
            on = a[1]
            for ch in self.chans.values():
                ch.a4_on = on
            self.push(t + (c['a4_on'] if on else c['a4_off']), ADV, ('a4toggle', not on))

    # ------------------------------------------------------------------ NS events
    def ev_epoch(self, t):
        c = self.cfg
        T_e = c['T_e']
        self.push(t + T_e, EPOCH)
        inw = self.in_window(t)
        attacked = c['a4_load'] > 0 and all(ch.a4_on for ch in self.chans.values())
        for k, ch in self.chans.items():
            carry = max(0.0, ch.busy_until - t)
            busy = max(ch.busy - carry, 0.0)
            ch.busy = carry
            cs = self.cas.epoch(k, busy, T_e, t)
            G_phy = cs['G_phy']
            if self.amc is not None:
                M_old = ch.mode
                M, p, T_g = self.amc.epoch(k, G_phy, cs['G_auth'], cs['m_hat'], cs['phi_G'],
                                           cs['T_c'], cs['S'])
                if M != M_old:
                    self.cas.set_mode(k, M, ch.devs)
                    if M == 'pure':
                        for i in ch.devs:
                            self.devs[i].tracking = False
                ch.mode, ch.p = M, p
                ch.T_slot = ch.T_air + T_g
            if inw:
                self.st['epochs'] += 1
                self.st['level_epochs'][cs['S']] += 1
                if attacked:
                    self.st['attacked_epochs'] += 1
                    self.st['attacked_level_epochs'][cs['S']] += 1
                if self.amc is not None:
                    self.st['p_sum'] += ch.p
                    self.st['p_n'] += 1
            if c['trace']:
                self.trace.append((t, k, ch.mode, ch.p, cs['S'], cs['T_c'], cs['Psi'], G_phy, cs['G_auth']))

    def ev_beacon(self, t):
        c = self.cfg
        self.push(t + c['beacon_period'], BEACON)
        for k, ch in self.chans.items():
            if ch.mode != 'slotted':
                continue
            synced = []
            for i in ch.devs:
                dv = self.devs[i]
                if not dv.tracking:
                    continue
                self.energy(dv, phy.P_RX_W * ch.T_beacon, t)
                dv.slot_err = self.rng.gauss(0, c['sigma_s'])
                dv.o_sync = (dv.delay - ch.d_bar) + self.rng.uniform(-c['eps_b'], c['eps_b'])
                dv.t_sync = t
                dv.t_beacon = t
                dv.sync_due = False
                dv.eps_cur = c['eps_b']
                synced.append(i)
            self.cas.on_beacon(k, t, synced)

    # ------------------------------------------------------------------ reservation rounds
    def ev_round(self, k, t):
        """Start of a reservation round: beacon, then TW request slots."""
        c, ch = self.cfg, self.chans[k]
        ch.contenders, ch.requests = [], []
        t_req0 = t + ch.T_beacon
        for i in ch.devs:
            dv = self.devs[i]
            if dv.state != 0 or not dv.queue or dv.t_dc > t_req0:
                continue
            if dv.cur is None:
                fr = Frame(i, k, dv.T_air, dv.prx, 'data', dv.queue[0])
                fr.c = dv.fcnt
                dv.fcnt += 1
                dv.cur = fr
            self.energy(dv, phy.P_RX_W * ch.T_beacon, t)
            # Sift-like reverse-exponential slot choice: most nodes pick late slots
            s = c['TW'] - min(int(self.rng.expovariate(1.0 / (c['sift_mean'] * c['TW']))), c['TW'] - 1)
            rq = Frame(i, k, ch.T_req, dv.prx, 'req')
            rq.slot = s
            rq.t_tx = t_req0 + (s - 1) * ch.T_req
            rq.rs = rq.t_tx + dv.delay
            rq.re = rq.rs + ch.T_req
            self.energy(dv, phy.P_TX_W * ch.T_req, rq.t_tx)
            self.push(rq.rs, RXS, rq)
            self.push(rq.re, RXE, rq)
            dv.state = 2
            ch.contenders.append(i)
        self.push(t_req0 + c['TW'] * ch.T_req + ch.d_max + 1e-3, REQEND, k)

    def ev_reqend(self, k, t):
        """End of the request window: grant up to n_grant decoded requests (VCTS),
        then contention-free data slots in grant order."""
        c, ch = self.cfg, self.chans[k]
        t_vcts_end = t + ch.T_ack
        granted, seen = [], set()
        for rq in sorted(ch.requests, key=lambda f: f.slot):
            if rq.dev in seen:
                continue
            seen.add(rq.dev)
            granted.append(rq.dev)
            if len(granted) >= c['n_grant']:
                break
        for i in ch.contenders:
            dv = self.devs[i]
            self.energy(dv, phy.P_RX_W * ch.T_ack, t)
            if i not in granted:
                dv.state = 0
        t_next = t_vcts_end
        for n, i in enumerate(granted):
            self.push(t_vcts_end + n * ch.T_slot, TX, i)
            t_next = t_vcts_end + (n + 1) * ch.T_slot
        self.push(t_next + ch.d_max + 1e-3, ROUND, k)

    # ------------------------------------------------------------------ results
    def results(self):
        st = dict(self.st)
        c = self.cfg
        acc = st['accepted']
        st['pdr'] = acc / st['generated'] if st['generated'] else 0.0
        st['energy_per_frame_mJ'] = 1000.0 * st['energy'] / acc if acc else float('nan')
        st['latency_mean'] = st['latency'] / acc if acc else float('nan')
        st['p_mean'] = st['p_sum'] / st['p_n'] if st['p_n'] else float('nan')
        st['collision_rate'] = st['collided'] / st['attempts'] if st['attempts'] else 0.0
        st['false_stale_rate'] = st['false_stale'] / st['legit_checked'] if st['legit_checked'] else 0.0
        st['a2_rate_acc'] = st['a2_acc'] / st['a2_att'] if st['a2_att'] else float('nan')
        st['a1_rate_acc'] = st['a1_acc'] / st['a1_att'] if st['a1_att'] else float('nan')
        st['relay_rate_acc'] = st['relay_acc'] / st['relay_att'] if st['relay_att'] else float('nan')
        st['level_changes'] = sum(cs.level_changes for cs in self.cas.ch.values())
        victims = [dv for dv in self.devs if dv.victim]
        st['relay_victims'] = len(victims)
        st['relay_flagged'] = sum(1 for dv in victims if self.cas.dev[dv.id].flagged)
        st['n_channels'] = len(self.chans)
        st['hours'] = (c['T_meas'] + c['T_tail'] + c['T_warm']) / 3600.0
        if self.amc is not None:
            st['mode_switches'] = sum(cc.switches for cc in self.amc.ch.values())
            st['slotted_share'] = (sum(sum(1 for m in cc.M_hist if m == 'slotted') for cc in self.amc.ch.values())
                                   / max(1, sum(len(cc.M_hist) for cc in self.amc.ch.values())))
        else:
            st['mode_switches'] = 0
            st['slotted_share'] = 1.0 if c['scheme'] == 'slotted' else 0.0
        st['sf_hist'] = {}
        for dv in self.devs:
            key = f"{dv.link}{dv.sf}"
            st['sf_hist'][key] = st['sf_hist'].get(key, 0) + 1
        st['a2_bins'] = self.a2_bins
        st['relay_log'] = self.relay_log
        st['trace'] = self.trace
        return st
