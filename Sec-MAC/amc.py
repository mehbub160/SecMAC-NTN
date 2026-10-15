"""AMC: Adaptive MAC Controller (network-server side, per logical channel, per epoch).

Algorithm 2 of the paper.  Inputs: observed load G_phy (from the CAE), the
authenticated load G_auth and attempts per frame m_hat (from CASAM's
accounting), the load trust phi_G, the context trust T_c and the security
level S.  Outputs: access mode M in {pure, slotted}, transmission probability
p and the slot period, broadcast to the devices of the channel.
"""
import math

DEFAULT_AMC = dict(
    theta_c=0.30,     # collision probability (pure-ALOHA equivalent) that triggers slotted access
    Delta_c=0.15,     # hysteresis band of the mode decision
    n_m=3,            # dwell: epochs the decision must persist before a switch
    eta_m=0.15,       # bound on the change of p per epoch
    theta_T=0.60,     # minimum trust for a mode switch
    p_min=0.05,
    G_star=dict(pure=0.5, slotted=1.0),   # throughput-maximising total load of each mode
    g0=0.02,
    trust_aware=True, mode_select=True, admission=True, stability=True,
    sigma_s=0.002,    # slot timing error of beacon-tracking devices (one sigma)
    duty=0.01,
)


class ChanCtrl:
    __slots__ = ('M', 'p', 'u', 'N', 'T_air', 'Delta', 'T_g', 'switches', 'p_hist', 'M_hist')

    def __init__(self, N, T_air, Delta, p0, mode):
        self.M, self.p, self.u = mode, p0, 0
        self.N, self.T_air, self.Delta = N, T_air, Delta
        self.T_g = 0.0
        self.switches = 0
        self.p_hist, self.M_hist = [], []


class Amc:
    def __init__(self, cfg: dict, chans: dict, fixed_mode=None):
        self.cfg = dict(DEFAULT_AMC)
        self.cfg.update(cfg or {})
        c = self.cfg
        self.ch = {}
        for k, v in chans.items():
            # worst case before any demand is known: every device uses its full duty cycle
            p0 = min(1.0, c['G_star']['pure'] / max(v['N'] * c['duty'], 1e-9)) if c['admission'] else 1.0
            mode = fixed_mode or 'pure'
            cc = ChanCtrl(v['N'], v['T_air'], v['Delta'], p0, mode)
            cc.T_g = 6 * c['sigma_s'] + 2 * v['Delta']
            self.ch[k] = cc
        self.fixed_mode = fixed_mode

    def epoch(self, k, G_phy, G_auth, m_hat, phi_G, T_c, S, available=True):
        c = self.cfg
        cc = self.ch[k]
        if not available:
            return cc.M, cc.p, cc.T_g
        # 1. trusted demand
        if c['trust_aware']:
            G_tr = G_auth + phi_G * (G_phy - G_auth)
        else:
            G_tr = G_phy
        G_tr = max(G_tr, 0.0)
        # collisions involve every frame on the air, so the mode follows the observed
        # load; an adversary can at most push a channel into the conservative mode
        P_c = 1.0 - math.exp(-2.0 * max(G_phy, G_tr))
        # 2. access mode with hysteresis and dwell; one-directional under alert
        if c['mode_select'] and self.fixed_mode is None:
            M_new = cc.M
            if cc.M == 'pure' and P_c > c['theta_c']:
                M_new = 'slotted'
            elif cc.M == 'slotted' and P_c < c['theta_c'] - (c['Delta_c'] if c['stability'] else 0.0):
                M_new = 'pure'
            # a move towards the conservative mode (slotted) is always allowed; a move
            # back to pure access needs sufficient trust and is frozen at S = 2
            if M_new == 'pure' and c['trust_aware']:
                allowed = T_c >= c['theta_T'] and S < 2
            else:
                allowed = True
            if M_new != cc.M and allowed:
                cc.u += 1
            else:
                cc.u = 0
            if cc.u >= (c['n_m'] if c['stability'] else 1):
                cc.M, cc.u = M_new, 0
                cc.switches += 1
        # 3. transmission probability steering the trusted load to the target of the mode
        if c['admission']:
            eta = c['eta_m']
            if c['trust_aware']:
                eta *= T_c
                if S >= 1:
                    eta *= 0.5
            if not c['stability']:
                eta = 1.0
            target = c['G_star'][cc.M]
            p_des = cc.p * target / G_tr if G_tr > c['g0'] else 1.0
            step = max(-eta, min(eta, p_des - cc.p))
            cc.p = max(c['p_min'], min(1.0, cc.p + step))
        cc.p_hist.append(cc.p)
        cc.M_hist.append(cc.M)
        return cc.M, cc.p, cc.T_g
