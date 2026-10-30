# AISimulate webpages

This directory owns webpage HTML and its assets. The public build is explicitly
allowlisted by `scripts/pages/build_pages_site.py`; a new directory is not published
automatically. The deployed paths remain stable across the move from docs.

- `index.html`: site landing page.
- `e2e-accuracy/`: E2E accuracy overview and retained historical snapshot.
- `fpm-accuracy/`: daily FPM accuracy overview; generated data comes from Actions.
- `fpe-support-matrix/`: current forward-pass support matrix.
- `feature-support-matrix/`: serving techniques per engine, rendered from
  `features.toml`.
- `support-matrix/`: legacy compatibility matrix.
- `universe/`: unpublished architecture explorer, retained for local use.

Prose documentation remains in docs. The Rust design HTML remains at
`crates/core/perfmodel/docs/design_doc.html` and is not deployed.
Older release branches still store E2E snapshots under the original docs path;
the publisher reads both locations without copying release HTML or JavaScript.
Missing snapshot paths are allowed; unreadable Git refs or objects fail the
publication check instead of bypassing the prior-snapshot comparison.

FPE qualification uploads the `fpe-nightly-web` data bundle produced by
`scripts/fpe/run_release_fpe.py`. Pages combines its qualified data with the HTML
and assets in `pages/fpe-support-matrix/`.
Retained qualified bundles may use the former `aiconfigurator_core` package
path or the current `aisimulate_core` path. Pages requires exactly one indexed
layout and validates its qualification and CSV identities before serving the
same public URLs. It never combines files from both layouts.

The Feature Support Matrix is static HTML that
`scripts/pages/feature_support_matrix.py` renders from
`feature-support-matrix/template.html` and `features.toml`. A pull request that
adds, removes or restricts a feature updates the affected cells in
`features.toml` and the composition boundaries in `docs/replay/features.md`;
each cell states what that branch's configuration validators accept. The
publisher renders main's file and each release branch's committed file as one
view per branch. A release cut before the file existed has no view.
Every row's `doc` page and anchor must exist in the same tree. Invalid main data
fails the build; an invalid release file only drops that release's view with a
build warning, so check a release-branch edit with main's builder before merging.

Build an empty output directory and serve it locally:

```bash
python scripts/pages/build_pages_site.py --output-dir /tmp/aisim-pages
python -m http.server --directory /tmp/aisim-pages 8000
```
