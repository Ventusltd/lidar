# Dharma: the shared rules

The shared rules of the federated repositories: one set of rules, kept in each repository, so that any session (local or Claude Code in the cloud) works the same way whichever repository it opens.

**This repository:** `lidar`, role: **data pipeline**. It holds free national LiDAR to 1 m bare-earth terrain tiles.

## The federation

| repository | visibility | role |
|---|---|---|
| [worlds-](https://github.com/Ventusltd/worlds-) | public | GPU maths, the furnace and the pair method |
| [graphics-engines-open-source](https://github.com/Ventusltd/graphics-engines-open-source) | public | explorer and public releases |
| site-world | private | 3D site design |
| [ventus-grid-engine](https://github.com/Ventusltd/ventus-grid-engine) | public | grid intelligence |
| [spiders](https://github.com/Ventusltd/spiders) | public | inspect and describe |
| [kuiper-drawing-engine](https://github.com/Ventusltd/kuiper-drawing-engine) | public | drawing |
| [particle-physics-drawing-engine](https://github.com/Ventusltd/particle-physics-drawing-engine) | public | drawing |
| [lidar](https://github.com/Ventusltd/lidar) | public | data pipeline |
| [soil](https://github.com/Ventusltd/soil) | public | data pipeline |

The same map, machine-readable, is `federation.json` beside this file.

## Shared rules

1. **Pair method.** Every number that matters is worked two independent ways (the electron and the positron) and compared against a stated tolerance. Pairs outside the tolerance are counted as photons and reported, never hidden. A CPU witness recomputes a sample. Agreement between two things we built shows consistency, not truth: tie results to cited worked examples, measured data or real engineers' checks before calling them verified.
2. **Receipts.** Every run writes a receipt: the inputs (or their hashes), the method, tolerances, photons, witness, timing, the hardware, and a sha256 of the canonical body. Readers and agents quote receipts; they do not recompute or paraphrase the numbers.
3. **Cartridges.** A feature is a small pure module plugged into a shared substrate: no page state, no network, one job, its own closed-form tests. One owner per cartridge at a time; nobody edits another's files.
4. **No names.** Public files never carry the names of clients, projects, sites, people or supplier brands, nor document or drawing numbers, local drive paths, account names or session links. Test land is anonymous (open-land-NN). Private material stays in private storage and private repositories.
5. **Licences.** Respect every licence and attribution (for example OGL v3.0, ODbL). Standards are cited by clause, never copied. Paid documents are used privately to calibrate and check; only our own method code and generic lessons are published.
6. **Measured words.** Estimates are labelled as estimates. Failures are shown with the advice that fixes them. Say what the data supports (for example 'measured to 1 m'), never 'exact' or 'as it is today'.
7. **Moderation.** Heavy numeric work runs on the GPU as scripts; agents read receipts and reason. Use the least compute that answers the question. Cap CI matrices so deploys are not starved.
8. **Releases.** Work on branches; run the gate and the pre-push hygiene checks before any push; merge through reviewed pull requests. Public releases are dated folders that are never edited after freezing.

## Working from the cloud

1. Clone the repository, read this file and `federation.json`, then the README.
2. Work on a branch. Run the repository's own tests and hygiene checks before pushing.
3. Open a pull request with what changed, the test results and the hygiene results. The owner merges.
4. If a task needs private material, stop and ask: it never comes into a public repository.
