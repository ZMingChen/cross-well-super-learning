# Release checklist

All items must be checked before a journal submission. A local folder or ZIP is not a substitute for a public repository.

- [ ] Authors selected a real software license and committed the license file.
- [ ] Public repository URL resolves without authentication.
- [ ] A tagged release and archive DOI were created; the exact URL and DOI are recorded in the manuscript.
- [ ] `python scripts/run_demo.py` succeeds in a clean virtual environment.
- [ ] `python scripts/repository_audit.py` returns `PASS`.
- [ ] README documents installation, input schema, expected outputs, seed, and limitations in English.
- [ ] No raw LAS, manual-label workbook, well list or identifier, source path, private email, point-level label/prediction, trained model artifact, or temporary file is tracked; `.gitignore` and `repository_audit.py` both pass.
- [ ] The manuscript's Computer Code Availability section contains the permanent repository URL.
- [ ] The cover letter contains the same URL.
- [ ] The data statement explains the confidential-data restriction and the public synthetic demonstration.
- [ ] A second author independently follows the README and confirms the output files are produced.
