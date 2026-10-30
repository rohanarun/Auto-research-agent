# Validation of this release

The initial research cohort is complete: 18 core cells (integer multiplication plus five kernels, three models) and 12 additional mathematical trials. Independent review remains pending for van der Waerden and Sidorenko; those six cells are unscored. The result data preserves this distinction.

- All 30 published records pass source/checkpoint/timing consistency checks. Exact sealed kernel sources, paired measurements, mathematical proof hashes and integer retiming witnesses are bound to their result records.
- All 30 harness tests pass in a prepared macOS workspace. They include positive local computation and compilation, denied external/localhost networking, denied curator/credential/other-workspace reads, symlink and hard-link boundaries, native-tool rejection, capacity retries, API action contracts, and GPU cleanup on failure. Test credentials and provider calls are synthetic.
- Clean setup retrieved all ten pinned snapshots and verified their content hashes. This clean-snapshot check used `--no-dependencies`; research trials used the separately prepared Python environment with the pinned dependencies.
- The tool-free Codex transport and confined local computation were tested with a real authenticated client. API adapters have local HTTP contract tests; they were not used for the initial research scores.
- The actual H200 worker passed unprivileged CUDA and network-boundary preflight before final scoring. Final scoring evaluated every public timed shape on two fresh banks for each numerically correct source. The compilation-failing attention candidate stopped on its first bank. These are local protocol measurements, not official OKC tests.
- The public provider adapter's read/upload/download operations were checked on an already allocated benchmark worker. The portable GPU orchestration uses the tested worker components and mocked lifecycle regressions; a second paid research cohort was not run just to exercise the portable refactor.
- Every recorded H200 lease has confirmed deletion. The [resource receipt](data/gpu-lifecycle.json) describes observed control-plane lifetime, not billed time or GPU utilization.

The initial private prototype and this portable release have different source hashes. Initial and replacement-cohort hashes are in [cohort metadata](data/cohort.json); the initial CLI is identified there as well. The release makes paths/provider configuration portable and fixes transport diagnostics and capacity retries. It does not retroactively change trial outcomes or the frozen scoring rule.

To rerun local checks after completing setup and exporting the workspace variables in the README:

```sh
python3.12 -m unittest discover -s tests -v
python3.12 tools/check_results.py
```

The first command needs the prepared command environment and macOS Seatbelt. The second checks the committed results without a model account or GPU.

## Baseline-speedup rescore

At the user’s correction, v2 uses measured speedup >1.01× as the primary kernel cutoff and records time to the exact tested source. The original results/policy are archived. No raw source, timing bank, interval, checkpoint or actor history was changed, and no model or GPU work was repeated. Two added policy tests check the strict cutoff and separation of timing support. AST comparison flags unchanged-baseline controls.
