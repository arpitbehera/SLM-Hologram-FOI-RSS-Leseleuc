# Contributing to foitweezers

Thanks for your interest in contributing. This project reproduces the numerical
results of Nishimura/Ando/de Léséleuc, *Phys. Rev. A* **113**, 013119 (2026).
Contributions that improve fidelity to the paper, performance, test coverage, or
documentation are all welcome.

Please read the [README](README.md) first — it explains the algorithm, the
backends, and the CLI. This guide covers the mechanics of contributing.

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproduction (preset, backend,
  command line, and the wrong vs. expected output).
- **Bug fixes** — small, focused PRs with a regression test.
- **Features** — the Phase 2 items in the README (vector-Debye VP criterion,
  non-square lattices, truncated-Gaussian sweep) are the prioritized roadmap.
  Open an issue to discuss design before large work.
- **Docs** — corrections, clarifications, and worked examples.

## Development setup

The project uses [`uv`](https://docs.astral.sh/uv/) and a `src/` layout.

```bash
git clone https://github.com/arpitbehera/SLM-Hologram-FOI-RSS-Leseleuc.git
cd SLM-Hologram-FOI-RSS-Leseleuc

uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'        # core + pytest
uv pip install -e '.[dev,torch]'  # add the torch backend (optional)
```

Verify the setup before changing anything:

```bash
pytest -q                              # unit + gradient + autodiff tests
python scripts/fig3.py --preset tiny   # quick CPU smoke (minutes)
```

## Making changes

1. **Branch off `main`.** Use a descriptive name, e.g. `fix/foi-gradient-sign`
   or `feat/non-square-lattice`.
2. **Keep PRs focused.** One logical change per PR. Unrelated cleanups belong in
   separate PRs.
3. **Match the surrounding style.** Follow the naming, structure, and comment
   density already in `src/foitweezers/`. No new linter is enforced; just keep it
   consistent.
4. **Preserve the faithful path.** The `scipy` backend is the reference
   reproduction. Changes to the forward model, cost functions, or gradients must
   keep it physically correct (energy-conserving FFT, analytic gradient matching
   finite differences — see the README's Algorithm section).

## Testing

All changes must keep the suite green:

```bash
pytest -q
```

- **Bug fixes** need a test that fails before the fix and passes after.
- **New cost functions / gradients** need a finite-difference gradient check
  (see existing tests in `tests/`), and a torch autodiff cross-check if a torch
  loss is added.
- **New CLI arguments** should be exercised with a `--preset tiny` smoke run; do
  not commit generated artifacts from `outputs/` (it is gitignored).

If you change numerical behavior, state in the PR which preset/backend you ran
and whether results still match the paper at `--preset paper` (or that you could
not run the GPU preset).

## Commit messages

This repo follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

- **type**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
- **scope**: the area touched, e.g. `optimize_mask`, `forward`, `readme`.
- **subject**: imperative, lowercase, ≤ ~50 chars, no trailing period.

Examples from this project's history:

```
feat(optimize_mask): best FOI/RSS mask generator CLI
docs(readme): document optimize_mask.py outputs and CLI args
```

Add a body only when the *why* is not obvious from the subject. Reference issues
with `Closes #NN` in the body when applicable.

## Pull requests

1. Push your branch and open a PR against `main`.
2. **Describe the change**: what, why, and how you verified it (commands run +
   output). Link any related issue.
3. **Confirm the checklist** in your PR description:
   - [ ] `pytest -q` passes locally
   - [ ] new/changed behavior has tests
   - [ ] README/docs updated if CLI, outputs, or algorithm changed
   - [ ] no generated files from `outputs/` committed
4. Keep the PR up to date with `main` (rebase preferred over merge commits).
5. Address review feedback in follow-up commits; maintainers may squash on merge.

## Reporting fidelity issues

Because this is a reproduction, "the numbers don't match the paper" is a valid
report. Include: the figure/table, the preset and backend, the spacing, the seed
count, and the observed vs. expected values. Note that absolute Table I numbers
only converge to the paper at `--preset paper` on a GPU — preset-independent
quantities are the `d/r_A` ratios.

## Questions

Open an issue with the `question` label, or start a discussion. Thanks for
helping reproduce the science.
