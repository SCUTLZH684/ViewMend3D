# ViewMend3D development

Follow `docs/optimization-framework.md` for the current optimization round.
The completed v1 protocol is archived there; continued optimization follows the preregistered `docs/optimization-v2.md`. Keep v1 results and default method semantics intact.
Keep the pinned original pipeline separate from controlled experiments.
Do not obtain candidate ground-truth RGB-D, masks or mesh errors during planning.
Use exact persisted common prefixes, explicit RNG domains, and recorded source versions.
Do not call an update event a new observation in the refinement-only control.
Do not claim quality improvements from CPU tests, synthetic fixtures or a single smoke run.
Run appropriate scoring/protocol/artifact/interface checks before reporting completion.
Use only genuinely idle GPUs; never stop other users' processes.
Keep datasets, maps, raw training frames, caches, SSH credentials and keys out of Git.
