# Efficient SLM Hologram Algorithms for Optical Tweezer Arrays Near the Rayleigh Limit

**Deep research brief — cited**
**Date:** 2026-05-18

---

## Executive Summary

Computer-generated phase-only holograms on liquid-crystal-on-silicon (LCoS) spatial light modulators (SLMs) are the workhorse for producing programmable arrays of optical tweezers, especially for neutral-atom quantum processors. The dominant *efficient* algorithm in production is the **Weighted Gerchberg–Saxton (WGS)** scheme of Di Leonardo, Ianni & Ruocco (2007) [1], often combined with **Mixed-Region Amplitude Freedom (MRAF)** (Pasienski & DeMarco, 2008) [2], camera-in-the-loop (CITL) feedback, and **Zernike-mode aberration correction** (Ebadi *et al.* 2021) [3]. These methods now drive demonstrated arrays from > 100 sites (Schymik 2020) [4] to 1,180 qubits (Atom Computing 2023) [5] and 6,100 atoms in ~12,000 SLM-generated traps (Manetsch *et al.* 2024) [6].

When inter-trap spacing approaches the Rayleigh distance $r_R = 1.22\lambda/(2\,\mathrm{NA})$, vanilla WGS uniformity collapses because all spots are mutually *coherent*: trap amplitudes interfere rather than just summing. Three lines of work address this:

1. **Coherent-forward-model optimization** — Ando *et al.* (arXiv:2411.03564) [7] reformulate hologram design as nonlinear optimization of a hand-crafted cost over the full phase pattern, achieving a 5×5 spot array at **0.952 µm pitch with NA 0.75 at λ = 820 nm** (≈ 1.43 × $r_R$), and propose a refined Rayleigh criterion for coherent spot arrays.
2. **High-fidelity feedback** — Chew *et al.* (arXiv:2407.20699; *PRA* 110:053518) [8] report **σ_I = 0.3 %**, shape variation < 0.5 %, and < 70 nm positional accuracy.
3. **Beam-shape engineering** — *super-oscillation* traps (Singh 2023 [9]; Nagar 2019 [10]) produce sub-Airy hotspots but at the cost of bright sidelobes that constrain atom-array applicability.

In software, practitioners primarily use **slmsuite** (Holodyne, ~158 ⭐, MIT) which implements GPU-accelerated GS/WGS/MRAF with CITL and Zernike correction [11]. Recent (2024–2025) algorithmic trends are (a) generative neural networks for near-instant hologram inference (Kim/Jo 2024, arXiv:2401.06014) [12] and (b) SLM-only parallel atom rearrangement via linear phase interpolation (Sotirova 2025, arXiv:2501.01391) [13], beginning to compete with the standard SLM-static + acousto-optic-deflector (AOD)-mobile architecture.

---

## 1. Problem Setup

A phase-only SLM modulates the wavefront $\phi(x,y)$ of a coherent beam. The far-field intensity is $|\mathcal{F}\{A_0(x,y)\,e^{i\phi}\}|^2$, where $A_0$ is the incident amplitude. The hologram-design problem is the **inverse**: given a target intensity $|U_T(u,v)|^2$ (an array of $N$ tweezers at positions $\{(u_n,v_n)\}$), find $\phi$ that produces it under hardware constraints (8-bit quantization, finite pixel count, dead-space, and known aberrations).

Two definitions of "efficient" matter:

- **Optical efficiency** $\eta$ — fraction of incident power delivered to the desired traps.
- **Computational efficiency** — wall-clock time per hologram, set by the number of FFT iterations.

A third axis, **uniformity** $\sigma_I = \mathrm{std}(I_n)/\langle I_n\rangle$, governs whether atom-loading rates are equal across the array.

The diffraction limit sets a *spatial* efficiency: any algorithm must produce traps separable by the imaging optics. For incoherent imaging the Rayleigh criterion is $r_R = 1.22\lambda/(2\,\mathrm{NA}) = 0.61\lambda/\mathrm{NA}$. For *coherent* arrays this must be refined — see §3.

---

## 2. Algorithm Families

### 2.1 Prisms-and-lenses superposition (RS / SR)
Early holographic tweezer arrays summed analytic phase ramps and Fresnel lens phases per trap, with optional random phase offsets, as documented in the Grier-group reviews [14]. Fast (no iteration) but uniformity is typically poor and significant power goes into ghost orders. Still used as the WGS initialization.

### 2.2 Gerchberg–Saxton (GS)
The original iterative Fourier-transform phase-retrieval algorithm dates to Gerchberg & Saxton, *Optik* 35:237 (1972), and is the foundation of essentially all subsequent SLM hologram methods [11, link to GS PDF therein]. Each iteration: FFT to far-field, replace amplitude with target keeping phase, inverse FFT, replace SLM-plane amplitude with the measured beam profile keeping phase. For *spot arrays* GS stagnates because the equal-weighting target does not penalize per-spot non-uniformity.

### 2.3 Weighted Gerchberg–Saxton (WGS) — production standard
Di Leonardo, Ianni & Ruocco, *Opt. Express* 15(4):1913 (2007) [1]. At iteration $k$ the target amplitude at spot $n$ is multiplied by $w_n^{(k+1)} = w_n^{(k)}\,\langle I\rangle/I_n^{(k)}$, so dim spots are amplified and bright spots are damped. The paper shows that all spot-array algorithms maximize a *performance quantifier* and introduces a new quantifier with provably better convergence [1]. WGS converges in tens of iterations to percent-level uniformity and is the default in `slmsuite` [11] and in essentially every neutral-atom-array experiment surveyed below.

### 2.4 Direct Search and Adaptive-Additive (AA)
Direct search (Yao/Wright/Roichman/Grier, "Optimized holographic optical traps") [15] flips one pixel at a time and accepts a change only if a cost decreases — slow but highest fidelity and best ghost-trap suppression. AA mixes computed and target amplitudes by a coefficient $\alpha$ and accelerates GS-style convergence; it is documented in the slmsuite algorithm reference [11] alongside the Bauschke / Soifer literature.

### 2.5 Mixed-Region Amplitude Freedom (MRAF)
Pasienski & DeMarco, *Opt. Express* 16:2176 (2008; arXiv:0712.0794) [2]. The Fourier plane is partitioned into a **signal region** (target enforced) and a **noise region** (algorithm free to dump intensity). Sacrificing $\eta$ in the signal region buys percent-level fractional error inside it; the abstract reports out-performance of "most frequently used alternatives" by roughly an order of magnitude on a fractional-error metric for arbitrary atom-trap shapes [2]. MRAF is implemented as a feedback variant in `slmsuite` [11].

### 2.6 Compressed-sensing variants
Pozzi & Bragheri introduced a compressed-sensing GS/WGS that updates only a subset of points each iteration, enabling real-time 3D point-cloud holography on a single GPU (arXiv:2003.05293) [16]; code at `ppozzi/SLM-3dPointCloud` [17]. An MDPI 2023 paper proposes an improved GS for holographic tweezers along similar lines [18].

### 2.7 Camera-in-the-loop (CITL)
A measured camera image of the focal plane replaces the simulated forward model in the weight update, removing systematic errors. Chew *et al.* (arXiv:2407.20699; *PRA* 110:053518) [8] combine CITL with multi-stage refinement and atom-light-shift metrology, reaching **σ_I = 0.3 %**, shape variation < 0.5 %, and < 70 nm positional precision.

### 2.8 Generative neural networks and differentiable optics
Kim/Jo *et al.* (arXiv:2401.06014) [12] trained a generative network that maps target spot pattern → SLM phase and demonstrated loading of cold strontium atoms into NN-generated arrays without per-shot WGS re-optimization. Differentiable forward models (PyTorch/JAX) are listed as planned in slmsuite [11] and underlie modern coherent-cost optimization (§2.9). Reviews of deep learning for optical tweezers more broadly are available in *Nanophotonics* 2024 [19].

### 2.9 Coherent-cost nonlinear optimization (Ando *et al.* 2024/26) — directly on-topic
Ando *et al.* (arXiv:2411.03564) [7] build a coherent forward model that explicitly includes the inter-spot interference that breaks WGS at small pitch, and optimize a hand-crafted nonlinear cost on the phase pattern (Adam-style). The abstract reports a 5×5 spot array at **0.952(1) µm pitch** with NA 0.75 at λ = 820 nm — overcoming "the limitation of a few micrometres under similar conditions" [7]. For these parameters the conventional Rayleigh distance is $0.61 \times 820\,\mathrm{nm} / 0.75 \approx 0.667\,\mathrm{µm}$, so the demonstrated pitch is ≈ 1.43 × $r_R$. The authors additionally propose a **refined Rayleigh criterion** that accounts for spot coherence and the trap separation/pitch ratio, supporting a "super-resolution" claim relative to the naive 1.22 λ/(2 NA) bound [7].

---

## 3. Behavior Near the Rayleigh Limit

### 3.1 Why naive WGS fails as pitch → $r_R$
Vanilla WGS enforces only the **diagonal** target intensities at the spot centers. Once Airy disks overlap, the focal-plane field is a coherent sum and the cross-terms between spots produce fringes inside the target spots and outside them. Per Ando *et al.* [7]:

> "as the spot interval approaches the wavelength of light, interference effects among the spots become prominent, which complicates the generation of a distortion-free alignment."

Observed effects (qualitative, traceable to [7] and to the Grier-group reviews [14]):
- Trap-trap interference fringes → non-uniformity grows.
- Ghost orders and "leaked" intensity between traps.
- Reduced trap depth/stiffness when overlapping Airy disks pull intensity from a target spot.

### 3.2 Refining the Rayleigh criterion for coherent spot arrays
The classical Rayleigh criterion is derived for *incoherent* point sources. For an SLM-generated tweezer array all spots are mutually coherent. Ando *et al.* [7] refine the criterion by accounting for the *visibility* and *separation* of the coherent spot superposition, and report "super-resolution" against the conventional 1.22 λ/(2 NA) bound under that refined definition. Note that this is a *definitional* refinement — the textbook incoherent criterion is intact; the relevant criterion for coherent tweezer arrays is simply different from the classroom one.

### 3.3 What helps near the Rayleigh limit
1. **MRAF** — give the algorithm freedom to dump intensity outside the target spots [2].
2. **Coherent forward-model nonlinear optimization** — Ando-style cost over the full coherent field, made tractable by differentiable / autodiff frameworks [7, 11].
3. **Camera-/atom-in-the-loop feedback** — eliminates systematic miscalibration. Chew 2024 reports σ_I = 0.3 %, shape variation < 0.5 %, < 70 nm position errors [8].
4. **Zernike aberration correction** — Ebadi *et al.* (Nature 595:227, 2021) [3] add Zernike polynomials to the SLM phase and tune each amplitude to maximize the *atom light shift*. The Glasgow group earlier built an SLM-based Shack–Hartmann wavefront sensor for the same purpose [20].
5. **Beam-shape engineering** — super-oscillation traps (Singh *et al.* 2023, *Commun. Phys.* 6:170 [9]; Nagar *et al.* 2019, *Opt. Lett.* 44:2430 [10]; review in *Nat. Rev. Phys.* 2021 [21]) produce sub-Airy hotspots but with bright sidelobes that limit atom applicability.
6. **Pixel-pitch oversampling and high-NA optics** — precondition: the SLM bandlimit must support the required spatial frequencies in the back aperture (general principle stated implicitly across [1, 2, 7]).

### 3.4 Quantitative state of the art (smallest pitches demonstrated)
- **Ando *et al.* (arXiv:2411.03564) [7]:** 5×5 grid, **0.952(1) µm** pitch, NA 0.75, λ = 820 nm. **Lowest SLM-generated tweezer pitch I located in this survey.**
- Chew *et al.* (arXiv:2407.20699) [8]: pitch not quoted in abstract; emphasis on intensity and positional precision (σ_I = 0.3 %, < 70 nm).
- Typical neutral-atom-array spacings reported elsewhere are in the few-µm range. Exact per-experiment spacings for Endres 2016 [22], Barredo 2016 [23], Ebadi 2021 [3], Manetsch 2024 [6] were not extracted in this brief — recovering them requires reading the main-text PDFs, which the workflow flagged as out of scope.

---

## 4. State of the Art in SLM-Generated Tweezer Arrays

### 4.1 Experimental milestones (verified from abstracts / press pages)

| Year | Group | Demonstration | Source |
|---|---|---|---|
| 2016 | Endres et al. (Lukin/Greiner) | 1D defect-free atom array, 51 atoms, atom-by-atom assembly | Science 354:1024 / arXiv:1607.03044 [22] |
| 2016 | Barredo et al. (Browaeys/Lahaye) | 2D defect-free atom assembly | Science 354:1021 / hal-01616802 [23] |
| 2020 | Schymik et al. | > 100-site fully-loaded arbitrary geometries | PRA 102:063107 / arXiv:2011.06827 [4] |
| 2021 | Ebadi et al. | 256-atom programmable simulator; Zernike SLM aberration correction | Nature 595:227 [3] |
| 2023 | Atom Computing | 1,225-site array, 1,180 qubits, first commercial > 1000 | [5, 24, 25] |
| 2024 | Bluvstein et al. | "Logical quantum processor", 48 logical qubits | Nature 626:58 / arXiv:2312.03982 [26] |
| 2024 | Manetsch et al. (Endres) | 6,100 atoms in ~12,000 SLM tweezer sites — largest neutral-atom array to date | arXiv:2403.12021 / Nature 647:60 (2025) [6] |
| 2024 | Lin et al. (Atom Computing) | Continuous operation of large lattice-tweezer arrays (¹⁷¹Yb) | arXiv:2402.04994 / arXiv:2401.16177 [27, 28] |
| 2024 | Chew, Poitrinal et al. (de Léséleuc) | "Ultra-precise" SLM array: σ_I = 0.3 %, < 70 nm positioning | arXiv:2407.20699 / PRA 110:053518 [8] |
| 2024 | Kim/Jo et al. | Generative-NN SLM patterns loading Sr atoms | arXiv:2401.06014 [12] |
| 2024 | Bradley et al. (Microsoft/Atom Computing) | Logical computation on neutral-atom processor | arXiv:2411.11822 [29] |
| 2024/26 | Ando et al. | Near-Rayleigh nonlinear-cost holograms; refined criterion | arXiv:2411.03564 [7] |
| 2025 | Sotirova et al. | SLM-only parallel atom rearrangement via linear phase interpolation | arXiv:2501.01391 [13] |

A metasurface (not SLM) demonstration of 78,400 traps in a single static element (arXiv:2512.08222) [30] is noted for context only — it is not an SLM-reconfigurable system.

### 4.2 Software ecosystem
- **slmsuite** [11] — Holodyne, MIT, Python + CUDA (cupy). Implements GS, WGS, MRAF (feedback), Zernike-basis aberration correction, camera feedback, spot-specific hologram classes.
- **ppozzi/SLM-3dPointCloud** [17] — GPU CS-GS/WGS for 3D point clouds (arXiv:2003.05293) [16].
- **crisbour/PhaseRetrieval** [31] — CUDA GS demo.

### 4.3 Hardware
LCoS phase-only SLMs from Meadowlark, Hamamatsu, Holoeye, Jasper Display dominate the field (general field knowledge; per-experiment vendor assignment not extracted to keep claims source-traceable). Pixel pitches in the few-µm range, 8-bit phase quantization, LCoS refresh ≤ ~60 Hz are typical. Diffraction efficiency typically tens of percent in practice.

### 4.4 Algorithmic trends 2023–2026
1. Coherent forward-model nonlinear-cost optimization explicitly for near-Rayleigh arrays [7].
2. Generative networks / differentiable optics for near-instant hologram inference [12, 11, 19].
3. Camera-/atom-in-the-loop feedback as the precision driver [8, 3].
4. SLM-only rearrangement via fast phase updates / linear interpolation [13], competing with SLM + AOD.
5. Continued scaling: 256 (2021) → 1,180 (2023) → 6,100 (2024) [3, 5, 6].

---

## 5. Disagreements and Caveats

- **"Super-resolution" framing** [7]: Ando *et al.* claim "super-resolution" by *refining* the Rayleigh criterion for coherent spot interference. This is a *definitional* refinement, not a violation of diffraction; the textbook incoherent 1.22 λ/(2 NA) bound is intact.
- **Uniformity vs. spacing trade-off**: Chew 2024 [8] achieves σ_I = 0.3 % but does not quote sub-µm pitches; Ando 2024 [7] achieves 0.95 µm pitch but does not quote σ_I in the abstract. The two SOTA frontiers — *fidelity* and *spacing* — have not been demonstrated together in a single paper at the time of this survey.
- **SLM vs. AOD vs. metasurface**: AODs remain dominant for fast atom rearrangement; SLMs dominate the static tweezer array; metasurfaces [30] can produce vastly larger static arrays but lose dynamic reconfigurability. SLM-only rearrangement via phase interpolation [13] is a recent competitor to SLM+AOD hybrids.
- **Super-oscillation traps** [9, 10, 21] beat the diffraction limit for the central spot but carry intense sidelobes — useful for some particles, problematic for atoms because the rings cause light shifts and heating.

---

## 6. Open Questions

1. Can coherent-cost optimization [7] reach sub-percent intensity uniformity *and* near-Rayleigh pitches simultaneously?
2. Can a generative network [12] be conditioned on a coherent forward model so its output already respects sub-Rayleigh constraints?
3. How does SLM 8-bit phase quantization limit attainable σ_I at fixed pitch?
4. Can SLM-only rearrangement [13] replace AOD handoffs at the scale of 10⁴ traps given LCoS refresh limits?
5. For the largest reported arrays (6,100 atoms) [6], what is the measured trap pitch and how close is it to the Rayleigh limit in those optics?

---

## 7. Recommended Next Steps (for implementers)

1. Baseline with `slmsuite` [11] (GS/WGS/MRAF + CITL + Zernike).
2. For near-Rayleigh arrays, reproduce the Ando *et al.* coherent-cost optimization [7] in PyTorch/JAX — the FFT forward model is trivially differentiable.
3. Drive intensity uniformity below 1 % via the Chew *et al.* recipe [8]: multi-stage refinement + atom-light-shift metrology + careful CITL calibration.
4. Correct aberrations Zernike-mode-by-Zernike-mode in atom-light-shift feedback (Ebadi 2021 Extended Data Fig. 2) [3].
5. For high-rate reconfiguration without AOD handover, consider linear phase interpolation between hologram frames [13].

---

## Sources

[1] Di Leonardo, Ianni & Ruocco, "Computer generation of optimal holograms for optical trap arrays," *Opt. Express* 15(4):1913 (2007). https://opg.optica.org/oe/fulltext.cfm?uri=oe-15-4-1913

[2] Pasienski & DeMarco, "A high-accuracy algorithm for designing arbitrary holographic atom traps," *Opt. Express* 16:2176 (2008). arXiv:0712.0794. https://arxiv.org/abs/0712.0794

[3] Ebadi *et al.*, "Quantum phases of matter on a 256-atom programmable quantum simulator," *Nature* 595:227 (2021). https://www.nature.com/articles/s41586-021-03582-4 ; Zernike-correction figure: https://www.nature.com/articles/s41586-021-03582-4/figures/7

[4] Schymik, Lienhard, Barredo, Scholl, Williams, Browaeys & Lahaye, "Enhanced atom-by-atom assembly of arbitrary tweezer arrays," *Phys. Rev. A* 102:063107 (2020). arXiv:2011.06827. https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.063107

[5] Atom Computing, "Novel solutions for continuously loading large atomic arrays" (Jan 31, 2024). https://atom-computing.com/novel-solutions-for-continuously-loading-large-atomic-arrays/

[6] Manetsch *et al.*, "A tweezer array with 6100 highly coherent atomic qubits," arXiv:2403.12021 (2024); *Nature* 647:60 (2025). https://arxiv.org/abs/2403.12021 ; https://www.nature.com/articles/s41586-025-09641-4 ; Caltech news https://www.caltech.edu/about/news/caltech-team-sets-record-with-6100-qubit-array

[7] Ando *et al.*, "Optimization-based hologram design for fine optical tweezers array and extension of super-resolution criteria," arXiv:2411.03564 (v1 Nov 2024, v2 Jan 2026). https://arxiv.org/abs/2411.03564

[8] Chew, Poitrinal, Tomita, Kitade, Mauricio, Ohmori & de Léséleuc, "Ultra-precise holographic optical tweezers array," arXiv:2407.20699 (2024); *Phys. Rev. A* 110:053518 (2024). https://arxiv.org/abs/2407.20699 ; https://ui.adsabs.harvard.edu/abs/2024PhRvA.110e3518C/abstract

[9] Singh *et al.*, "Single atom in a superoscillatory optical trap," *Communications Physics* 6:170 (2023). https://www.nature.com/articles/s42005-023-01271-4

[10] Nagar *et al.*, "Optical trapping below the diffraction limit with a tunable beam waist using super-oscillating beams," *Opt. Lett.* 44(10):2430 (2019). https://opg.optica.org/abstract.cfm?uri=ol-44-10-2430

[11] slmsuite — Holodyne/community Python + CUDA package for SLM control and holography. GitHub: https://github.com/holodyne/slmsuite ; docs: https://slmsuite.readthedocs.io/ ; algorithms reference: https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.html ; PyPI: https://pypi.org/project/slmsuite/

[12] Rim/Kim/Jo *et al.*, "Creation of a tweezer array for cold atoms utilizing a generative neural network," arXiv:2401.06014 (2024). https://arxiv.org/abs/2401.06014

[13] Sotirova *et al.*, "Parallel assembly of neutral atom arrays with an SLM using linear phase interpolation," arXiv:2501.01391 (2025). https://arxiv.org/abs/2501.01391

[14] Grier-group reviews / Curtis, Koss & Grier, "Dynamic Holographic Optical Tweezers" and "Holographic optical trapping": https://physics.nyu.edu/grierlab/dynamic4c/ ; https://physics.nyu.edu/grierlab/aoreview3c/ ; https://physics.nyu.edu/grierlab/cgh3c/

[15] Yao/Wright/Roichman/Grier, "Optimized holographic optical traps": https://physics.nyu.edu/grierlab/optimal4c/

[16] Pozzi & Bragheri, "Real time computer generation of three-dimensional point cloud holograms through GPU implementation of compressed sensing Gerchberg–Saxton algorithm," arXiv:2003.05293 (2020). https://arxiv.org/abs/2003.05293

[17] ppozzi/SLM-3dPointCloud — GPU GS/WGS/CS-GS Python: https://github.com/ppozzi/SLM-3dPointCloud

[18] "Holographic Optical Tweezers That Use an Improved Gerchberg–Saxton Algorithm," *Micromachines* 14(5):1014 (2023). https://www.mdpi.com/2072-666X/14/5/1014

[19] "Deep learning for optical tweezers," *Nanophotonics* (2024). https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2024-0013/html ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11502085/

[20] "An SLM-based Shack–Hartmann wavefront sensor for aberration correction in optical tweezers" — Glasgow eprint: https://eprints.gla.ac.uk/67140/

[21] "Optical superoscillation technologies beyond the diffraction limit," *Nature Reviews Physics* (2021). https://www.nature.com/articles/s42254-021-00382-7

[22] Endres *et al.*, "Atom-by-atom assembly of defect-free one-dimensional cold atom arrays," *Science* 354:1024 (2016); arXiv:1607.03044. https://pubmed.ncbi.nlm.nih.gov/27811284/ ; http://arxiv.org/abs/1607.03044

[23] Barredo *et al.*, "An atom-by-atom assembler of defect-free arbitrary two-dimensional atomic arrays," *Science* 354:1021 (2016). https://www.science.org/doi/10.1126/science.aah3778 ; hal-01616802

[24] Ars Technica coverage of Atom Computing 1,180-qubit announcement (Oct 2023): https://arstechnica.com/science/2023/10/atom-computing-is-the-first-to-announce-a-1000-qubit-quantum-computer

[25] PRNewswire, "Quantum Startup Atom Computing First to Exceed 1,000 Qubits" (Oct 24, 2023): https://www.prnewswire.com/news-releases/quantum-startup-atom-computing-first-to-exceed-1-000-qubits-301964712.html

[26] Bluvstein *et al.*, "Logical quantum processor based on reconfigurable atom arrays," *Nature* 626:58 (2024); arXiv:2312.03982. https://www.nature.com/articles/s41586-023-06927-3 ; https://arxiv.org/abs/2312.03982

[27] "Continuous operation of large-scale atom arrays in optical lattices," arXiv:2402.04994. https://arxiv.org/html/2402.04994v2

[28] "Iterative assembly of ¹⁷¹Yb atom arrays in cavity-enhanced optical lattices," arXiv:2401.16177. https://arxiv.org/html/2401.16177v2

[29] "Logical computation demonstrated with a neutral atom quantum processor," arXiv:2411.11822. https://arxiv.org/abs/2411.11822v1

[30] "Direct Generation of an Array with 78400 Optical Tweezers Using a Single Metasurface," arXiv:2512.08222 (preprint; *metasurface* not SLM — included for context only). https://arxiv.org/html/2512.08222v1

[31] crisbour/PhaseRetrieval — CUDA GS demo. https://github.com/crisbour/PhaseRetrieval
