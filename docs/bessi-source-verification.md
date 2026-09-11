# BESSI source verification

Checked on 11 September 2026. The upstream college-student workbook was identified through the authors' [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8760475/), which links the Excel data at [OSF 7cktv](https://osf.io/7cktv/). The underlying research is [Soto et al. (2022), The BESSI](https://doi.org/10.1037/pspp0000401).

The OSF download response resolved to `https://files.osf.io/v1/resources/4zgyr/providers/osfstorage/611552c0d698660097a2c5b8`. Its `x-waterbutler-metadata` header identified:

- Filename: `College student sample.xlsx`.
- Version: 1; size: 605,359 bytes.
- Modified: `2021-08-12T16:56:32.613533+00:00`.
- SHA-256: `1424d3ccdbc6f1b8dd63bce66b4fb6c12e2525ff5c220b6b14383ad15391ff93`.

That hash matches the local workbook and the preparation manifest exactly. The raw file was not replaced or edited. The paper's 322-person college sample agrees with the source row count. This establishes provenance, but does not resolve the non-unique `Case` field or independently verify participant identity for our row-level split.

Both the [project API](https://api.osf.io/v2/nodes/4zgyr/) and [parent API](https://api.osf.io/v2/nodes/dyr97/) returned `node_license: null`. Record the dataset's redistribution terms as unspecified; do not transfer an article's licence to the workbook. Share source links and code rather than raw workbook data. The separately downloaded automated scoring template was not hash-verified during this check, and the original local download date remains unknown.

Read-only verification commands:

```bash
curl -sS -L -I https://osf.io/7cktv/download
shasum -a 256 'data/raw/bfi2-bessi/College student sample.xlsx'
curl -sS https://api.osf.io/v2/nodes/4zgyr/
curl -sS https://api.osf.io/v2/nodes/dyr97/
```
