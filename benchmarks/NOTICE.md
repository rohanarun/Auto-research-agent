# Source provenance and ownership

Benchmark runner, documentation and chart code are contributed under the parent repository's MIT license. Upstream research manuscripts, problem statements, starter kernels, reference checks and baseline implementations remain the property of their respective authors and retain their original terms; the parent MIT license does not relicense them.

The five public MUSE problem packages were obtained read-only from Open Kernel Challenge, identified by the problem IDs and SHA-256 manifests in `data/snapshot-hashes`. Their SGLang baselines identify the SGLang source commit in their headers. The benchmark makes no official submission and does not represent the challenge organizer's leaderboard.

Mathematical task setup fetches pinned source trees from the repositories identified in `scenarios.json`. Source-manifest dependencies are fetched at their pinned upstream commit and checked against the recorded content hashes. Future-improvement trees stay in the curator area and never enter an actor snapshot.

Private optimized kernel implementations are excluded. Published candidate code is newly generated during this benchmark; it must not be confused with those private reference implementations.
