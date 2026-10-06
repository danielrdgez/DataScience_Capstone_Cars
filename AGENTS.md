# AGENTS

## Project Instructions

- Work only on the `main` branch unless the user explicitly asks for another branch.
- Do not create branches with the `codex/` prefix for this project unless the user explicitly asks.
- Read the project files before making changes, especially `README.md`, `DATA_DICTIONARY.md`, and relevant scripts.
- Keep changes focused on the user's request and avoid unrelated refactors.
- Preserve user-created work. Do not revert or overwrite changes unless the user explicitly asks.
- Update `README.md` when project structure, setup steps, or workflows change.
- Update `DATA_DICTIONARY.md` when datasets, fields, sources, or transformations change.
- Whenever a project change affects documented behavior, update `README.md` and/or `DATA_DICTIONARY.md` in the same change: keep the README current with project workflows and use Mermaid diagrams there when they clarify a workflow; keep the data dictionary limited to data-specific information about the dataset selected for model development.
- Whenever a project process or workflow changes, update `README.md` in the same change with the current process description and a Mermaid diagram of that process and workflow.
- Whenever a planned or completed data change affects the model-development dataset (including its sources, fields, schema, or transformations), update `DATA_DICTIONARY.md` in the same change so its data definitions and profiles reflect that change.
- Use clear commit messages that describe the change.

## Data Scientist Working Standards

- Work as a professional data scientist: use sound methodology, explicit data definitions, reproducible analysis, and evidence-based conclusions.
- Write clean, concise, legible code with descriptive names and focused functions. Keep comments brief and use them only to explain non-obvious reasoning or constraints.
- Keep file creation proportional to the task. Reuse existing modules and workflows; avoid excessive test files, redundant scripts, and unnecessary scaffolding.
- Verify consequential transformations, joins, leakage controls, and model behavior with focused checks. Prefer a small number of meaningful tests over tests that simply mirror the implementation.

## Project Notes

This repository supports a data science capstone project about cars. The project will contain datasets, NHTSA-related resources, YouTube-related resources, exploratory data analysis, and machine learning models.

## Cohort Definition

- Whenever the user refers to a vehicle "cohort," interpret it as the combination of `ModelYear`, `Make`, `Model`, and `Trim` from the `NHTSA_vPIC` dataset.
- Keep cohort dimensions as separate columns in datasets, summaries, and tables. A combined cohort label may be used for chart display only.
- Link used-car listings to `NHTSA_vPIC` using its lowercase `vin` column when calculating listing price or mileage statistics for cohorts. Never use the capitalized `VIN` decoder-variable column as the join key.
