"""Frame format, message integrity code and field reconstruction of SecMAC-NTN.

The simulator does not run AES for every frame (it would dominate the run
time); it checks the *semantics* of the MIC: a frame passes iff it was built
under the device's session key AND the receiver reconstructs the same 32-bit
counter and 32-bit stamp that the sender bound into the MIC.  The functions
below implement the real computation with AES-128-CMAC, and
tests/test_casam.py verifies that the abstract check and the real check give
identical verdicts, including the reconstruction corner cases.

Frame fields on the air (beyond LoRaWAN 1.0.4):
  * FCnt      : 16 low bits of the 32-bit uplink counter c  (LoRaWAN field)
  * stamp cmd : proprietary MAC command, 4 bytes = CID | tau20 (20 bits) |
                j (3 bits) | sync flag (1 bit)
      tau20 = low 20 bits of the device time at emission in milliseconds
      j     = attempt index of this counter value (0 = first transmission)
      sync  = 1 if the device applied a DeviceTimeAns since its last frame
MIC = AES-CMAC_K(B0' | msg)[0..3], where B0' is the LoRaWAN B0 block with the
32-bit counter and, in the four bytes that LoRaWAN leaves at 0x00, the full
32-bit stamp.  Receivers reconstruct c and tau before verifying the MIC, so a
wrong reconstruction fails the MIC exactly like a forged frame.
"""
import struct

try:
    from cryptography.hazmat.primitives import cmac
    from cryptography.hazmat.primitives.ciphers import algorithms
    _HAVE_CRYPTO = True
except Exception:  # pragma: no cover
    _HAVE_CRYPTO = False

TAU_BITS = 20
TAU_MOD = 1 << TAU_BITS
CNT_MOD = 1 << 16
STAMP_CMD_BYTES = 4


def b0_block(dev_addr: int, counter32: int, stamp32: int, msg_len: int) -> bytes:
    """LoRaWAN B0 block (uplink, Dir=0) with the stamp in the four reserved bytes."""
    return (bytes([0x49]) + struct.pack('<I', stamp32 & 0xFFFFFFFF) + bytes([0x00])
            + struct.pack('<I', dev_addr & 0xFFFFFFFF) + struct.pack('<I', counter32 & 0xFFFFFFFF)
            + bytes([0x00, msg_len & 0xFF]))


def compute_mic(key: bytes, dev_addr: int, counter32: int, stamp32: int, msg: bytes) -> bytes:
    if not _HAVE_CRYPTO:
        raise RuntimeError("cryptography package not available")
    c = cmac.CMAC(algorithms.AES(key))
    c.update(b0_block(dev_addr, counter32, stamp32, len(msg)) + msg)
    return c.finalize()[:4]


def encode_stamp_cmd(tau32: int, j: int, sync_flag: int, cid: int = 0x80) -> bytes:
    v = ((tau32 & (TAU_MOD - 1)) << 4) | ((j & 0x7) << 1) | (sync_flag & 1)
    return bytes([cid]) + v.to_bytes(3, 'big')


def decode_stamp_cmd(b: bytes):
    v = int.from_bytes(b[1:4], 'big')
    return v >> 4, (v >> 1) & 0x7, v & 1


def reconstruct_counter(c16: int, c_max: int) -> int:
    """32-bit counter whose low 16 bits are c16, closest to the last accepted
    counter c_max.  LoRaWAN requires the receiver to track the high bits."""
    base = c_max & ~0xFFFF
    cands = [base + c16, base + c16 + CNT_MOD, base + c16 - CNT_MOD]
    cands = [c for c in cands if 0 <= c < (1 << 32)]
    return min(cands, key=lambda c: abs(c - c_max))


def reconstruct_stamp(tau20: int, expected_ms: float) -> int:
    """32-bit stamp whose low 20 bits are tau20, closest to the expected device
    time (ms).  Unambiguous while |error| < 2^19 ms (about 524 s)."""
    e = int(round(expected_ms)) & 0xFFFFFFFF
    base = e & ~(TAU_MOD - 1)
    cands = [base + tau20, base + tau20 + TAU_MOD, base + tau20 - TAU_MOD]
    return min(cands, key=lambda t: abs(t - e))
