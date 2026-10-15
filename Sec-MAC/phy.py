"""Physical-layer helpers: LoRa airtime, link budget, SF assignment, geometry.

All numbers that enter the paper are defined here with their source.
"""
import math

C_LIGHT = 299_792_458.0          # m/s
R_EARTH = 6_371_000.0            # m

# SX1276 datasheet Rev. 7 (May 2020), Table "LoRa receiver sensitivity",
# 125 kHz bandwidth, 862-1020 MHz band.
SENSITIVITY_DBM = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -133.0, 12: -136.0}

# SX1276 datasheet: TX 29 mA at +13 dBm (RFO), RX 11.5 mA (LnaBoost on), VDD 3.3 V.
P_TX_W = 0.029 * 3.3             # 95.7 mW
P_RX_W = 0.0115 * 3.3            # 38.0 mW
TX_POWER_DBM = 13.0


def airtime(sf: int, payload_bytes: int, bw_hz: float = 125e3, cr: int = 1,
            n_preamble: int = 8, explicit_header: bool = True, crc: bool = True) -> float:
    """LoRa time on air in seconds (Semtech SX1276 datasheet, Sec. 4.1.1.7).

    payload_bytes is the PHY payload (the complete LoRaWAN frame including MIC).
    Low data rate optimisation is enabled for SF11 and SF12 at 125 kHz.
    """
    de = 1 if sf >= 11 else 0
    h = 0 if explicit_header else 1
    t_sym = (2.0 ** sf) / bw_hz
    t_pre = (n_preamble + 4.25) * t_sym
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) - 20 * h
    den = 4 * (sf - 2 * de)
    n_pay = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return t_pre + n_pay * t_sym


def fspl_db(d_m: float, f_hz: float) -> float:
    return 20 * math.log10(d_m) + 20 * math.log10(f_hz) - 147.55


def terrestrial_pathloss_db(d_m: float, shadow_db: float) -> float:
    """Log-distance model of Petajajarvi et al. (ITST 2015), 868 MHz, on ground:
    PL(d) = 128.95 + 10*2.32*log10(d/1 km) dB, shadowing sigma 7.8 dB."""
    return 128.95 + 10 * 2.32 * math.log10(max(d_m, 1.0) / 1000.0) + shadow_db


def choose_sf(prx_dbm: float, margin_db: float):
    """Lowest SF whose sensitivity is met with the given margin, or None."""
    for sf in range(7, 13):
        if prx_dbm >= SENSITIVITY_DBM[sf] + margin_db:
            return sf
    return None


def slant_range_m(ground_dist_m: float, h_m: float) -> float:
    """Slant range from a ground point at central angle (ground distance) to a
    satellite at altitude h over the sub-satellite point."""
    psi = ground_dist_m / R_EARTH
    r_sat = R_EARTH + h_m
    return math.sqrt(R_EARTH ** 2 + r_sat ** 2 - 2 * R_EARTH * r_sat * math.cos(psi))


def ground_radius_for_elevation_m(elev_deg: float, h_m: float) -> float:
    """Ground distance from the sub-satellite point at which the elevation
    angle equals elev_deg."""
    th = math.radians(elev_deg)
    r_sat = R_EARTH + h_m
    psi = math.acos(R_EARTH * math.cos(th) / r_sat) - th
    return psi * R_EARTH


def ntn_delay_bounds_s(h_m: float, elev_min_deg: float, feeder_m: float):
    """One-way device -> satellite -> NTN gateway delay bounds for a transparent
    payload (service link plus a fixed feeder link)."""
    d_near = h_m
    d_far = slant_range_m(ground_radius_for_elevation_m(elev_min_deg, h_m), h_m)
    return (d_near + feeder_m) / C_LIGHT, (d_far + feeder_m) / C_LIGHT
