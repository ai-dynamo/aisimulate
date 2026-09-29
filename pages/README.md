# AISimulate webpages

This directory owns webpage HTML and its assets. The public build is explicitly
allowlisted by `scripts/build_pages_site.py`; a new directory is not published
automatically. The deployed paths remain stable across the move from docs.

- `index.html`: site landing page.
- `e2e-accuracy/`: E2E accuracy overview and retained historical snapshot.
- `fpm-accuracy/`: daily FPM accuracy overview; generated data comes from Actions.
- `fpe-support-matrix/`: current forward-pass support matrix.
- `support-matrix/`: legacy compatibility matrix.
- `universe/`: unpublished architecture explorer, retained for local use.

Prose documentation remains in docs. The Rust design HTML remains at
`crates/core/perfmodel/docs/design_doc.html` and is not deployed.
Older release branches still store E2E snapshots under the original docs path;
the publisher reads both locations without copying release HTML or JavaScript.
Missing snapshot paths are allowed; unreadable Git refs or objects fail the
publication check instead of bypassing the prior-snapshot comparison.

FPE qualification uploads the `fpe-nightly-web` data bundle produced by
`scripts/run_release_fpe.py`. Pages combines its qualified data with the HTML
and assets in `pages/fpe-support-matrix/`.
Retained qualified bundles may use the former `aiconfigurator_core` package
path or the current `aisimulate_core` path. Pages requires exactly one indexed
layout and validates its qualification and CSV identities before serving the
same public URLs. It never combines files from both layouts.

Build an empty output directory and serve it locally:

```bash
python scripts/build_pages_site.py --output-dir /tmp/aisim-pages
python -m http.server --directory /tmp/aisim-pages 8000
```
