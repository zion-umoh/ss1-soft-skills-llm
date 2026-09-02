# Model 2: Big Five and BESSI data intake

Model 2 learns a relationship from Big Five personality measures to selected
BESSI soft-skill measures. The current source files are under
`data/raw/bfi2-bessi/`; their recorded preparation choices are in
`data/metadata/model2-data-spec.template.json`.

## Required inputs

1. **Big Five measurements**
   - The questionnaire/version (for example, BFI-10, BFI-44, or already
     scored trait values).
   - Either one score column for each trait or the item columns, response
     range, and reverse-scoring key needed to calculate scores.

2. **BESSI measurements**
   - The exact BESSI skills/dimensions that Model 2 should predict.
   - For each selected skill, either a supplied score column or the item
     columns, response range, and reverse-scoring key needed to calculate it.

3. **Participant linkage**
   - A stable participant identifier shared by both datasets, or a documented
     one-to-one mapping if the identifier names differ.
   - If the measures are already in one file, identify its participant-ID
     column.

4. **Dataset provenance**
   - Source URL or provider, licence/permission, version, and download date.

## Preparation rules once the data is available

- Preserve the source files unchanged in `data/raw/model2/`.
- Validate IDs, response ranges, duplicate participants, missing values, and
  the expected questionnaire columns before scoring.
- Apply documented reverse scoring and compute trait/skill scores from the
  declared item lists only.
- Keep the participant ID for auditing and split assignment; exclude it from
  model features.
- Make one fixed, participant-level train/validation/test split using seed 42
  (70%/15%/15%), so no participant appears in more than one split.
- Produce model-ready tables whose feature columns are the five Big Five
  traits and whose target columns are the selected BESSI skills.
- Report row counts, missingness, score distributions, correlations, and the
  final feature/target schema before training.

## What cannot be assumed

The current workflow uses the supplied BFI-2 and BESSI **domain-scale**
columns rather than recomputing scores from individual items. The BESSI skills
to predict are the five supplied domains. The source's `Case` column is reused
across distinct measurement rows, so it is not a unique participant identifier
and does not support participant-level split claims.
