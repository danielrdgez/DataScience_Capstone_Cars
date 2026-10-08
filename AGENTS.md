# AGENTS

## Project Instructions

- Work on `main`; use another branch or a `codex/` branch only if explicitly requested.
- Read `README.md`, `DATA_DICTIONARY.md`, and relevant scripts before editing.
- Keep changes focused. Preserve user work; do not revert or overwrite it without an explicit request.
- Update documentation in the same change as the behavior it describes: `README.md` for structure, setup, and workflows; `DATA_DICTIONARY.md` for the selected model-development dataset only.
- Describe workflow changes concisely in `README.md` and include a compact Mermaid diagram.
- Keep dictionary definitions and profiles current with planned or completed changes to model-data sources, fields, schemas, or transformations.
- Use clear commit messages that describe the change.

## Data Scientist Working Standards

- Use sound methodology, explicit data definitions, reproducible analysis, and evidence-based conclusions.
- Write legible code with descriptive names, focused functions, and apparent logic. Comment only on non-obvious reasoning or constraints.
- Keep documentation compact and professional: short sentences, a few bullets, and useful tables. Avoid repeated explanations, boilerplate, and unnecessary narrative.
- Reuse existing code, tests, and documentation. Do not create a test file, Markdown file, report, or scaffold for every change; add files only when necessary.
- Verify consequential transformations, joins, leakage controls, and model behavior with a few meaningful checks. Avoid redundant tests or tests that mirror the implementation.

## Project Notes

Car data science capstone: listings, NHTSA data, YouTube comments, EDA, and price models.

## Cohort Definition

- A vehicle cohort is `ModelYear`, `Make`, `Model`, and `Trim` from `NHTSA_vPIC`.
- Keep these as separate columns; combined labels are for chart display only.
- Join listings to vPIC on lowercase `vin`, including for cohort price/mileage statistics. Never join on the capitalized decoder field `VIN`.
