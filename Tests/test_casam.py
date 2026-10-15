"""Unit tests of the CASAM acceptance logic (Algorithm 1), the AMC (Algorithm 2)
and the frame format with real AES-128-CMAC.  Run:  python3 tests/test_casam.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from secmac import crypto
from secmac.casam import Casam, FORGED, REPLAY, STALE, REACK, ACCEPT
from secmac.amc import Amc
from secmac.phy import airtime

KEY = bytes(range(16))
DEV_ADDR = 0x26011BDA


class FakeFrame:
    def __init__(self, dev, k, c, tau32, j=0, key_ok=True, sync_flag=0):
        self.dev, self.k, self.c, self.j, self.key_ok, self.sync_flag = dev, k, c, j, key_ok, sync_flag
        self.c16 = c & 0xFFFF
        self.tau32 = tau32 & 0xFFFFFFFF
        self.tau20 = self.tau32 & 0xFFFFF
        self.kind = 'data'


def make_casam(policy='tvw', **over):
    T_air = airtime(7, 54)
    chans = {0: dict(T_air=T_air, d_bar=0.0, Delta=0.0, d_max=1e-5, link='ter')}
    cfg = dict(policy=policy)
    cfg.update(over)
    cas = Casam(cfg, chans, 2, [0, 0])
    for d in cas.dev:
        d.t_sync = 0.0
    return cas, T_air


class TestCrypto(unittest.TestCase):
    def test_mic_roundtrip_and_reconstruction(self):
        """The receiver reconstructs c and tau from 16/20 bits and the real MIC verifies."""
        msg = b'\x40' + DEV_ADDR.to_bytes(4, 'little') + b'\x00' + b'payload-bytes'
        for c, c_max in ((0, -1), (5, 4), (65535, 65530), (65536, 65535), (70000, 69999), (131072, 131071)):
            for tau in (0, 123456, 1048575, 1048576, 2 ** 31 + 5):
                mic = crypto.compute_mic(KEY, DEV_ADDR, c, tau, msg)
                c_rec = crypto.reconstruct_counter(c & 0xFFFF, max(c_max, 0))
                tau_rec = crypto.reconstruct_stamp(tau & 0xFFFFF, tau + 300)   # expected within 300 ms
                self.assertEqual(c_rec, c)
                self.assertEqual(tau_rec, tau & 0xFFFFFFFF)
                self.assertEqual(crypto.compute_mic(KEY, DEV_ADDR, c_rec, tau_rec, msg), mic)

    def test_wrong_reconstruction_fails_mic(self):
        """A stamp reconstructed one wrap away, or a wrong key, changes the MIC."""
        msg = b'frame'
        tau = 2 ** 20 + 10
        mic = crypto.compute_mic(KEY, DEV_ADDR, 7, tau, msg)
        self.assertNotEqual(crypto.compute_mic(KEY, DEV_ADDR, 7, tau - 2 ** 20, msg), mic)
        self.assertNotEqual(crypto.compute_mic(KEY, DEV_ADDR, 7 + 2 ** 16, tau, msg), mic)
        self.assertNotEqual(crypto.compute_mic(bytes(16), DEV_ADDR, 7, tau, msg), mic)

    def test_stamp_cmd(self):
        b = crypto.encode_stamp_cmd(2 ** 32 - 1, 5, 1)
        self.assertEqual(len(b), 4)
        tau20, j, f = crypto.decode_stamp_cmd(b)
        self.assertEqual((tau20, j, f), (0xFFFFF, 5, 1))

    def test_abstract_check_matches_real_cmac(self):
        """The simulator's abstract MIC check (key_ok and exact reconstruction) gives
        the same verdict as the real CMAC for a batch of frames with random
        counters, stamps and expected ages."""
        import random
        rng = random.Random(1)
        msg = b'x' * 20
        for _ in range(300):
            c = rng.randrange(0, 200000)
            c_max = c - rng.randrange(0, 5) if rng.random() < 0.9 else c - rng.randrange(0, 2 ** 17)
            tau = rng.randrange(0, 2 ** 30)
            expected = tau + rng.uniform(-600, 600) * (1 if rng.random() < 0.9 else 2000)
            key_ok = rng.random() < 0.8
            key = KEY if key_ok else bytes(16)
            mic = crypto.compute_mic(key, DEV_ADDR, c, tau, msg)
            c_rec = crypto.reconstruct_counter(c & 0xFFFF, max(c_max, 0))
            tau_rec = crypto.reconstruct_stamp(tau & 0xFFFFF, expected)
            real_ok = crypto.compute_mic(KEY, DEV_ADDR, c_rec, tau_rec, msg) == mic
            abstract_ok = key_ok and c_rec == c and tau_rec == tau
            self.assertEqual(real_ok, abstract_ok)


class TestCasam(unittest.TestCase):
    def legit(self, cas, T_air, dev, c, t, offset=0.0, j=0, **kw):
        tau32 = int((t - T_air - offset) * 1000)
        return FakeFrame(dev, 0, c, tau32, j=j, **kw), t

    def test_order_forged_replay_stale_accept(self):
        cas, T_air = make_casam()
        t = 1000.0
        fr, tr = self.legit(cas, T_air, 0, 0, t, key_ok=False)
        self.assertEqual(cas.check(fr, tr), FORGED)
        self.assertEqual(cas.dev[0].c_max, -1)                      # state unchanged
        fr, tr = self.legit(cas, T_air, 0, 0, t)
        self.assertEqual(cas.check(fr, tr), ACCEPT)
        # exact copy of an accepted frame (A1) is a replay, whatever the window says
        self.assertEqual(cas.check(fr, tr + 0.001), REPLAY)
        # suppress-and-replay: unused counter, same stamp, one airtime later -> stale
        fr2, tr2 = self.legit(cas, T_air, 0, 1, t + 30.0)
        self.assertEqual(cas.check(fr2, tr2 + T_air + 0.005), STALE)
        self.assertEqual(cas.dev[0].c_max, 0)                       # counter still unused
        # the legitimate retransmission of that counter still passes
        fr3, tr3 = self.legit(cas, T_air, 0, 1, t + 40.0, j=1)
        self.assertEqual(cas.check(fr3, tr3), ACCEPT)
        # ACK lost: the device retransmits with the same counter and j+1 -> re-ack, no delivery
        fr4, tr4 = self.legit(cas, T_air, 0, 1, t + 50.0, j=2)
        self.assertEqual(cas.check(fr4, tr4), REACK)
        # a copy of attempt 1 after attempt 2 was seen is a replay
        self.assertEqual(cas.check(fr3, tr3 + 20.0), REPLAY)
        # an older counter is a replay
        self.assertEqual(cas.check(fr, tr + 100.0), REPLAY)

    def test_counter_policy_accepts_suppressed_replay(self):
        cas, T_air = make_casam(policy='counter')
        fr, tr = self.legit(cas, T_air, 0, 0, 1000.0)
        self.assertEqual(cas.check(fr, tr + T_air + 5.0), ACCEPT)     # no window: accepted late

    def test_window_tightens_with_profile_and_grows_with_cap(self):
        cas, T_air = make_casam()
        d = cas.dev[0]
        w0 = cas.window(0, 0, 100.0)
        self.assertAlmostEqual(w0, 0.100 + 20e-6 * 100.0 + 0.003)   # cap: eps_ts + drift + eps0
        fr, tr = self.legit(cas, T_air, 0, 0, 100.0, offset=0.020)
        self.assertEqual(cas.check(fr, tr), ACCEPT)
        w1 = cas.window(0, 0, 100.5)
        self.assertLess(w1, w0)                                       # profile is tighter than the cap
        self.assertGreater(w1, 0.020)                                 # but admits the device's offset
        # profile cannot be widened by a rejected frame
        fr2, tr2 = self.legit(cas, T_air, 0, 1, 130.0, offset=0.020)
        self.assertEqual(cas.check(fr2, tr2 + 1.0), STALE)
        self.assertAlmostEqual(cas.window(0, 0, 130.5), w1 + 20e-6 * 30.0, places=6)

    def test_sync_flag_resets_profile_to_cap(self):
        cas, T_air = make_casam()
        fr, tr = self.legit(cas, T_air, 0, 0, 100.0)
        cas.check(fr, tr)
        self.assertTrue(cas.dev[0].prof)
        cas.on_devtime_ans(0, 200.0)
        fr2, tr2 = self.legit(cas, T_air, 0, 1, 300.0, offset=0.080, sync_flag=1)
        self.assertEqual(cas.check(fr2, tr2), ACCEPT)                 # 80 ms offset admitted by the cap
        self.assertEqual(cas.dev[0].t_sync, 200.0)

    def test_sync_flag_commits_only_on_acceptance(self):
        """A flagged frame that is rejected (replayed counter, or stale) leaves the
        synchronisation state and the profile untouched; the legitimate flagged
        retransmission then commits the correction."""
        cas, T_air = make_casam()
        fr, tr = self.legit(cas, T_air, 0, 0, 100.0)
        cas.check(fr, tr)
        cas.on_devtime_ans(0, 200.0)
        d = cas.dev[0]
        before = (d.t_sync, d.prof, d.pending_sync_t, d.o_hat)
        fr, tr = self.legit(cas, T_air, 0, 0, 300.0, sync_flag=1)       # old counter: replay
        self.assertEqual(cas.check(fr, tr), REPLAY)
        self.assertEqual((d.t_sync, d.prof, d.pending_sync_t, d.o_hat), before)
        fr, tr = self.legit(cas, T_air, 0, 1, 300.0, offset=5.0, sync_flag=1)   # 5 s old: stale
        self.assertEqual(cas.check(fr, tr), STALE)
        self.assertEqual((d.t_sync, d.prof, d.pending_sync_t, d.o_hat), before)
        fr, tr = self.legit(cas, T_air, 0, 1, 400.0, offset=0.050, sync_flag=1)
        self.assertEqual(cas.check(fr, tr), ACCEPT)
        self.assertEqual(d.t_sync, 200.0)
        self.assertIsNone(d.pending_sync_t)

    def test_level_escalates_immediately_and_deescalates_with_hysteresis(self):
        cas, T_air = make_casam()
        c = cas.ch[0]
        # forged frames in one epoch -> trust drops -> level 1 or 2 at once
        for i in range(20):
            cas.check(FakeFrame(0, 0, i, 0, key_ok=False), 10.0 + i)
        cas.epoch(0, busy=5.0, T_e=60.0, t=60.0)
        self.assertGreaterEqual(c.S, 1)
        S_hi = c.S
        # quiet epochs: de-escalation needs n_d consecutive epochs above threshold + hysteresis
        levels = []
        for n in range(1, 12):
            cas.epoch(0, busy=5.0, T_e=60.0, t=60.0 * (n + 1))
            levels.append(c.S)
        self.assertEqual(levels[0], S_hi)                             # not immediately
        self.assertEqual(levels[-1], 0)                               # eventually
        self.assertTrue(all(a >= b for a, b in zip(levels, levels[1:])))

    def test_authenticated_accounting_counts_lost_frames(self):
        cas, T_air = make_casam(r_max=3)
        fr, tr = self.legit(cas, T_air, 0, 0, 100.0)
        cas.check(fr, tr)
        fr, tr = self.legit(cas, T_air, 0, 4, 400.0, j=2)            # counters 1..3 never arrived
        cas.check(fr, tr)
        ep = cas.ch[0].ep
        self.assertEqual(ep.f, 5)
        self.assertEqual(ep.a, 1 + (3 + 3 * 4))


class TestAmc(unittest.TestCase):
    def test_bounded_step_and_one_directional_freeze(self):
        amc = Amc({}, {0: dict(N=100, T_air=0.1, Delta=0.0)})
        cc = amc.ch[0]
        p0 = cc.p
        amc.epoch(0, G_phy=3.0, G_auth=3.0, m_hat=2.0, phi_G=1.0, T_c=1.0, S=0)
        self.assertGreaterEqual(cc.p, p0 - 0.15 - 1e-9)             # step bound
        # heavy observed load: switch to slotted after the dwell even at S = 2
        for _ in range(3):
            amc.epoch(0, G_phy=3.0, G_auth=3.0, m_hat=2.0, phi_G=1.0, T_c=0.3, S=2)
        self.assertEqual(cc.M, 'slotted')
        # light load but low trust / high level: no move back to pure
        for _ in range(5):
            amc.epoch(0, G_phy=0.02, G_auth=0.02, m_hat=1.0, phi_G=1.0, T_c=0.3, S=2)
        self.assertEqual(cc.M, 'slotted')
        for _ in range(5):
            amc.epoch(0, G_phy=0.02, G_auth=0.02, m_hat=1.0, phi_G=1.0, T_c=1.0, S=0)
        self.assertEqual(cc.M, 'pure')

    def test_trusted_demand_ignores_untrusted_load(self):
        amc = Amc({}, {0: dict(N=100, T_air=0.1, Delta=0.0)})
        cc = amc.ch[0]
        cc.p = 0.5
        amc.epoch(0, G_phy=2.0, G_auth=0.3, m_hat=1.5, phi_G=0.0, T_c=1.0, S=0)
        self.assertGreater(cc.p, 0.5)                                 # legit load below target: open up
        cc.p = 0.5
        amc.epoch(0, G_phy=2.0, G_auth=0.3, m_hat=1.5, phi_G=1.0, T_c=1.0, S=0)
        self.assertLess(cc.p, 0.5)                                    # trusted load above target: throttle


if __name__ == '__main__':
    unittest.main(verbosity=2)
