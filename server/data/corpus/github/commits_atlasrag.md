# GitHub Commit History: AtlasRAG

Repository: https://github.com/vanshnarang13/AtlasRAG
Last 5 commits (most recent first)

## Commit 72508990 — 07 Jul 2026
Author: Vansh Narang

Message: Fix ERD image not rendering properly on GitHub

The previous README embedded a temporary Eraser export URL with a
transparent background, which rendered cropped/invisible on GitHub.
Committing a properly exported (opaque background, full canvas) PNG
into the repo instead so it renders reliably.

Files changed (2):
  modified   README.md  (+1 -1)
  added      docs/assets/database-erd.png  (+0 -0)

## Commit 190ad154 — 07 Jul 2026
Author: Vansh Narang

Message: Add database ERD diagram to README

Replaces the plain-text data model list with a visual entity-relationship
diagram generated via Eraser, matching the current Supabase migration schema.

Files changed (1):
  modified   README.md  (+5 -9)

## Commit 540de9e8 — 15 Jun 2026
Author: vanshnarang13

Message: Add API Reference section heading

Files changed (1):
  modified   README.md  (+1 -0)

## Commit bff75e0c — 08 Apr 2026
Author: Vansh Narang

Message: README updated

Files changed (1):
  modified   README.md  (+0 -11)

## Commit 238a0c61 — 08 Apr 2026
Author: Vansh Narang

Message: updated frontend

Files changed (100):
  added      .DS_Store  (+0 -0)
  added      README.md  (+298 -0)
  added      client/.gitignore  (+41 -0)
  added      client/eslint.config.mjs  (+18 -0)
  added      client/next.config.ts  (+7 -0)
  added      client/package-lock.json  (+6791 -0)
  added      client/package.json  (+30 -0)
  added      client/postcss.config.mjs  (+7 -0)
  ... and 92 more files
