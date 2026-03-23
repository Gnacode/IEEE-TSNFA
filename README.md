# TSNFA: Temporal Spectral Noise-Floor Adaptation for IoT Mesh Networks

[![Paper](https://img.shields.io/badge/Paper-IEEE_IoT-blue)](https://doi.org/XXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Interactive Results](https://img.shields.io/badge/Results-Interactive_Dashboard-orange)](https://gnacode.github.io/IEEE-TSNFA/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gnacode/IEEE-TSNFA/blob/main/notebooks/TSNFA_Step_by_Step.ipynb)

Monte Carlo simulation framework comparing **Temporal Spectral Noise-Floor Adaptation (TSNFA)** against five established event detection algorithms for autonomous edge triggering in IoT mesh sensor networks.

> **S. Makovetskii and L. Thomsen**, "Temporal Spectral Noise-Floor Adaptation for Autonomous Edge Triggering in IoT Mesh Sensor Networks," *IEEE Internet of Things Journal*, 2026.

---

## Key Results

200-node mesh network, 24-hour simulation, realistic multi-component noise model:

| Method | Detection Rate | False Positives | Precision | Latency (ms) |
|--------|:-:|:-:|:-:|:-:|
| **TSNFA (Proposed)** | **100.0%** | **0** | **100.0%** | **23.5** |
| Zhang et al. 2023 | 73.4% | 919,842 | 1.5% | 33.3 |
| STFT (Bhoi et al. 2022) | 100.0% | 399,822 | 4.6% | 29.4 |
| DEDaR (Hussein et al. 2022) | 100.0% | 13,387,930 | 0.3% | 27.5 |
| SoD (Correa et al. 2019) | 0.0% | 0 | 0.0% | 0.0 |
| TinyML (Hammad et al. 2023) | 99.7% | 5,465,607 | 0.5% | 30.4 |

**TSNFA achieves 100% detection rate with zero false positives** by combining three defences: spectral band selection, temporal persistence filtering, and adaptive noise-floor tracking.

## Interactive Dashboard

Explore the simulation results interactively:

**[Launch Interactive Dashboard](https://gnacode.github.io/IEEE-TSNFA/)**

The dashboard includes:
- Per-method performance comparison with interactive bar charts
- Radar plot comparing all six metrics across methods
- Noise model explorer showing the signal composition
- Parameter sensitivity analysis

No installation required. Runs in any modern browser.

---
## Jupiter Notebook in CoLab

Run a simulations in CoLab and see the performance of each algorithm against each other:

**[Launch Jupiter Notebook in CoLab](https://colab.research.google.com/github/Gnacode/IEEE-TSNFA/blob/main/notebooks/TSNFA_Step_by_Step.ipynb)**
The simulation matches the results found by the Monte Carlo simulation without implementing the full 24 hour simulation. It can be used to compare the different models performance to each other. The full simulation and the graphics routine are found here in the repository.

## Repository Structure

```
tsnfa-iot-simulation/
├── README.md
├── LICENSE
├── simulation/
│   ├── IOTfulltest4-withNoise4.py    # Monte Carlo simulation engine
│   ├── requirements.txt              # Python dependencies
│   └── presets.md                    # Simulation preset configurations
├── visualization/
│   ├── SimVisu4.py                   # Publication figure generator
│   └── requirements.txt
├── results/
│   ├── sim_200_ACCURATE.json         # 200-node simulation results
│   └── figures/                      # Generated publication figures (PDF + PNG)
│       ├── fig1_dashboard.pdf
│       ├── fig2_hexagonal.pdf
│       ├── fig3_waveforms.pdf
│       ├── fig5_publication.pdf
│       └── fig6_radar.pdf
├── docs/
│   └── index.html                    # Interactive GitHub Pages dashboard
├── notebooks/
│   └── TSNFA_Step_by_Step.ipynb    # Interactive walkthrough (runs in Colab)
└── paper/
    ├── Section_III_Theoretical_Framework.md
    └── algorithms/                   # Formatted pseudocode (docx)
```

## Running the Simulation

### Requirements

- Python 3.10 or higher
- ~4 GB RAM for 200-node simulation
- ~2 hours computation time (200 nodes, 24h simulated, ACCURATE preset)

### Installation

```bash
git clone https://github.com/Gnacode/IEEE-TSNFA.git
cd tsnfa-iot-simulation
pip install -r simulation/requirements.txt
```

### Quick Start

Run the 200-node simulation with the ACCURATE preset:

```bash
cd simulation
python IOTfulltest4-withNoise4.py --nodes 200 --preset ACCURATE --output ../results/
```

The simulation will:
1. Generate a 24-hour synthetic signal with Poisson-distributed events
2. Apply the multi-component noise model (thermal + EMI + digital switching)
3. Run all six detection algorithms on identical signal realisations
4. Output JSON results and trigger logs

### Simulation Presets

| Preset | Nodes | Duration | Events/hr/node | Description |
|--------|:-----:|:--------:|:--------------:|-------------|
| QUICK | 50 | 1 hr | 1.0 | Fast validation (~2 min) |
| STANDARD | 100 | 12 hr | 0.5 | Medium fidelity (~20 min) |
| ACCURATE | 200 | 24 hr | 0.5 | Paper results (~2 hr) |

### Generating Figures

After running the simulation:

```bash
cd visualization
python SimVisu4.py --input ../results/ --output ../results/figures/
```

Generates all publication figures at 1200 DPI (PDF + PNG).

## Signal Model

The simulation evaluates all algorithms against a common synthetic signal:

```
x_i[n] = s[n - τ_i] + w_th[n] + w_EMI[n] + w_dig[n]
```

| Component | Description | Parameters |
|-----------|-------------|------------|
| `s[n]` | Event waveform | Damped sinusoidal impulse, 1-5 Hz, SNR = 18 dB |
| `w_th` | Thermal noise | White Gaussian, power spectral density P/fs |
| `w_EMI` | EMI interference | 60 Hz sine, amplitude 0.3√P |
| `w_dig` | Digital switching | Intermittent bursts, 800-2000 Hz, up to 2.0√P |

Noise power drifts sinusoidally: `P(t) = P₀ · 10^((6/10)·sin(2πt/3600))`, swinging between P₀/4 and 4P₀ over each hour.

## Algorithm Overview

TSNFA combines three defences that no other method implements simultaneously:

1. **Spectral band selection** — 128-point FFT isolates the 1-5 Hz event band, discarding EMI and digital noise
2. **Temporal persistence** — γd-frame mean/median filter rejects single-frame transients
3. **Adaptive noise floor** — EMA or median tracker follows the ±6 dB noise drift

See [Section III: Theoretical Framework](paper/Section_III_Theoretical_Framework.md) for complete algorithm descriptions with pseudocode and line-by-line annotations.

## Hardware Platform

The algorithms target the **STM32G071** (Arm Cortex-M0+ at 64 MHz, 36 KB SRAM, no FPU). TSNFA requires approximately 100 arithmetic operations per frame, well within the compute budget of this resource-constrained platform.

## Citation

```bibtex
@article{makovetskii2026tsnfa,
  title={Temporal Spectral Noise-Floor Adaptation for Autonomous Edge 
         Triggering in IoT Mesh Sensor Networks},
  author={Makovetskii, Sergii and Thomsen, Lars},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

## Related Work

This simulation study extends our earlier single-node validation:

> S. Makovetskii and L. Thomsen, "Temporal Spectral Noise-Floor Adaptation for Error-Intolerant Trigger Integrity in IoT Mesh Networks," 2026. [Paper](https://doi.org/YYYY)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **GNACODE INC.** — [gnacode.com](https://gnacode.com)
- Lars Thomsen — Medicine Hat, Alberta, Canada
- Sergii Makovetskii
