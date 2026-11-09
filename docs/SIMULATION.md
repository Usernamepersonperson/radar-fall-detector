# Simulation model and parameters

Everything in the results is produced by this model. It is a deliberately simple
physics model, not a validated radar channel model.

## Radar (`radarfall/config.py`)

| Parameter | Value | Derived |
|---|---|---|
| Carrier | 60 GHz | wavelength 5.0 mm |
| Sweep bandwidth | 1.5 GHz | range resolution 10 cm |
| Ramp time / chirp interval | 40 us / 250 us | slope 37.5 THz/s, PRF 4 kHz |
| ADC samples per chirp | 64 complex (IQ), 1.6 MHz | max range 6.4 m |
| Chirps per frame | 32 | velocity resolution 0.31 m/s |
| Max unambiguous speed | | +/- 5.0 m/s |
| Frame period | 50 ms (20 Hz) | one 8 ms burst per frame |
| RX channels | 2, stacked vertically, lambda/2 apart | elevation from phase |
| Mount | wall, 1.8 m high, boresight horizontal | azimuth FoV +/- 65 deg (cosine pattern) |

SNR is defined as the post-FFT SNR of a 1 m^2 scatterer at 3 m (nominal 30 dB).

## Signal synthesis (`radarfall/radar.py`)

For each frame the beat signal of K scatterers is summed per RX channel:
`s[rx, m, n] = sum_k a_k exp(j 4 pi (r_k + v_k m Tc) / lambda) exp(j 2 pi (r_k / dr) n / Ns) exp(j pi sin(el_k) rx)`

* amplitude `a_k ~ sqrt(RCS_k) / r_k^2` (two-way) times the azimuth pattern, with 12 % scintillation per frame
* velocity `v_k` is the finite-difference radial speed, held constant during the 8 ms burst
* multipath: a floor-bounce image (mirrored in z, gain 0.30) and a back-wall image
  (x mirrored at 5.2 m, gain 0.15) of every body scatterer
* clutter: 16 static reflectors, 2 swaying reflectors (curtain/fan, 2-6 mm at 1-3 Hz) which
  the install-time calibration masks out, 0.004 rad LO phase jitter on static clutter
* thermal noise: complex Gaussian, level set by the SNR definition above

Not modelled: range migration inside a burst, antenna coupling, real RCS aspect
dependence, furniture shadowing, several people, pets, real radar phase/amplitude
calibration errors, interference.

## Human model (`radarfall/motion.py`, `radarfall/activities.py`)

12 point scatterers: head, chest, abdomen, pelvis, 2 upper arms, 2 forearms,
2 thighs, 2 shanks (RCS 0.08 to 0.8 m^2). Pose = 12 numbers (position, heading,
whole-body tilt about the feet or pelvis, torso flexion, thigh/knee angles, arm
angles). Activities are keyframes with easing; walking adds a gait overlay;
breathing is a 4 mm chest oscillation. Falls use an accelerating tilt (`u^2.2`).

| Group | Activities |
|---|---|
| Falls (labelled, 7 kinds) | forward, backward, left, right (0.45-0.85 s), walking trip, from a chair, slow slide-down (2-3.6 s) |
| Fall-like, not falls | sit down (normal 1.6-2.6 s, fast 0.55-0.9 s), lie on bed (3-5 s) and flop onto bed (0.9-1.5 s), bend to pick up, squat to pick up, sit on the floor, deliberate lie on the floor (3-6 s), stumble and recover |
| Everyday | walk (0.45-1.2 m/s), stand, sit idle with fidgeting, sleep, leave the room |

Daily-living streams are random sequences of these (no falls). Fall scenarios are
6-12 s of standing/walking/sitting, one fall, then 32 s on the floor (30 % of the
time with small limb movements).
