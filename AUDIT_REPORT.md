# Repository audit following CAGEO-D-26-01542

## Verdict

The prior submission did not provide a functional public software repository. It provided a manuscript package with aggregate outputs and a short package README, but not a reproducible code release. This directly matches the desk-rejection reason concerning “the code repository/README.”

## Findings in the prior working directory

| Finding | Evidence | Submission risk |
|---|---|---|
| No public software project structure | Root had no software README, no dependency file, no license, and no release tag | Editors could not install or run the workflow |
| No reproducible environment | Core scripts import NumPy, pandas, SciPy, scikit-learn, Matplotlib, and optional XGBoost/LightGBM/CatBoost, but versions were undocumented | A clean environment could not resolve dependencies reliably |
| Actual runtime failure | In the audit environment, the core script failed at import because Matplotlib had been built against NumPy 1.x while NumPy 2.x was present; another entry point lacked SciPy | Confirms the repository was not functional from a new user's perspective |
| Hard-coded private inputs | Core scripts refer to local `outputs/tables/...` intermediate data products | Outside readers cannot supply inputs through a documented interface |
| Confidential material in the working tree | Full feature tables, well identifiers, depths, source files, and point-level predictions are present | Publishing the full work directory risks violating data-owner restrictions |
| README mismatch | The submitted README described upload-package contents, not code installation, input schema, commands, expected outputs, or limitations | It was not a “proper README” for a software repository |

## Corrective design now prepared

The `public_code_repository/` folder is a safe public-release scaffold. It supplies an English README, pinned environment, synthetic demo data, executable demo, sensitive-file guard, data schema, release checklist, and manuscript-ready code-availability text. It intentionally contains no real well data or point-level outputs.

## Remaining blocking actions

1. Authors must choose and add a real open-source license.
2. Run `python -m pip install -r requirements.txt` in a clean Python 3.10 or 3.11 virtual environment, then run `python scripts/release_preflight.py`.
3. Have a second author repeat the preflight independently.
4. Create a public GitHub/GitLab repository, push only the contents of `public_code_repository/`, and create a versioned release.
5. Archive that exact release with a DOI, then replace the placeholders in `CODE_AVAILABILITY.md` and the manuscript.
6. Before a new submission, verify the public URL anonymously in a browser and retain the passing preflight output.

## Non-negotiable boundary

The synthetic demonstration establishes that the published code can run; it does not reproduce the manuscript's numerical results. The manuscript must explicitly distinguish the public synthetic demo from the owner-restricted full-data analysis.
