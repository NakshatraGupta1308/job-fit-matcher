# job-fit-matcher

Scores how well a resume fits a job description, shows what is missing, and suggests bullet edits that make your resume speak the posting's language **without inventing experience**.

Everything runs locally. No API keys, no accounts, no data leaves your machine.

```
$ resume-matcher match my_resume.pdf backend_role.txt ml_role.txt -f text

JOB RANKING
-----------
 1.    70  [#######---]  Good fit     Senior Backend Engineer, Data Platform
 2.    12  [#---------]  Weak fit     Machine Learning Engineer, Computer Vision
```

## What you get

For each job description, a report with:

- **Overall match score (0 to 100)**, split into *requirement coverage* (how well your bullets address each requirement, by meaning) and *keyword coverage* (named skills from the posting that appear in your resume).
- **Per-requirement breakdown**: every requirement, whether it is required, a responsibility, or nice to have, its score, and the resume line that best supports it.
- **Gap analysis**: weakly covered requirements, missing required and nice-to-have skills, skills you list but never show in a bullet, other technical terms from the posting that are absent, and bullets that are probably irrelevant for this role.
- **Suggested edits**, each with the reasoning and a "check" reminder:
  - use the posting's spelling of a skill you already have ("Postgres" to "PostgreSQL")
  - lead with the action ("Responsible for migrating..." to "Migrated...")
  - name a tool your resume already lists when a bullet about that work leaves it out (flagged "only if true")
  - point at the closest existing bullet and the posting's words it does not use yet
  - honest notes when something is simply not in your resume, with no rewrite offered

See [`examples/sample_report.md`](examples/sample_report.md) for a full report.

### The no-fabrication rule

Suggestions are built only from what is already in the resume: the original bullet, skills the resume already mentions somewhere, and the job description's spelling of those same skills. A final guard throws away any suggested text that would name a skill the resume does not contain, and the test suite checks this on every run. When a requirement is genuinely missing, the tool says so and does not offer a rewrite.

## Install

Requires Python 3.9+.

```bash
git clone https://github.com/NakshatraGupta1308/job-fit-matcher.git
cd job-fit-matcher
python -m venv .venv && source .venv/bin/activate

pip install -e ".[embeddings,ui,dev]"   # everything
# or: pip install -r requirements.txt
```

The first run with the embeddings backend downloads the `all-MiniLM-L6-v2` model (about 90 MB) from Hugging Face and caches it. After that it works offline.

## Usage

### Command line

```bash
# One job, markdown report to stdout
resume-matcher match data/sample_resume.pdf data/sample_job.txt

# Save to a file, as plain text or JSON
resume-matcher match resume.pdf job.txt -f text -o report.txt
resume-matcher match resume.pdf job.txt -f json -o report.json

# Paste a job description from the clipboard (macOS example)
pbpaste | resume-matcher match resume.pdf -

# Compare many postings at once: one report per job plus a ranking
resume-matcher match resume.pdf jobs/*.txt --output-dir reports/

# Keep a history of scores across applications
resume-matcher match resume.pdf job.txt --save
resume-matcher history

# Inspect what the parsers extracted
resume-matcher parse-resume resume.pdf
resume-matcher parse-job job.txt
```

Useful options:

| Option | Meaning |
| --- | --- |
| `-f, --format` | `markdown` (default), `text`, or `json` |
| `-o, --output` | write a single report to a file |
| `--output-dir` | write one report per job plus `ranking.*` |
| `--backend` | `auto` (default), `sentence-transformers`, or `tfidf` |
| `--model` | any sentence-transformers model name or local path |
| `--max-suggestions` | how many suggestions per job (default 8) |
| `--save`, `--history-file` | append results to a local JSON lines history |

History lives in `~/.resume_matcher/history.jsonl` by default (override with `RESUME_MATCHER_HOME`). It is a plain text file you can read or delete.

### Web UI

```bash
streamlit run app/streamlit_app.py
```

Upload a resume (PDF, TXT, or MD) or paste it, paste one or more job descriptions, and get the scores, requirement table, missing keywords, suggestions, and markdown/text/JSON downloads. Several postings are ranked side by side.

### From Python

```python
from resume_matcher.analysis import analyze
from resume_matcher.report import render

result = analyze("resume.pdf", "job.txt")   # paths or raw text
print(result.score, result.result.fit_label)
print(render(result, "markdown"))
```

## How it works

```
resume (PDF/TXT/MD) --> resume_parser --> sections + bullets + skills -+
                                                                      +-> scorer --> gap_analysis --> rewrite_suggester --> report
job description ------> job_parser ----> weighted requirements ------+
```

1. **Parsing** (`src/resume_matcher/parser/`). Text is pulled from PDFs with `pdfplumber`, unicode punctuation is normalized, and lines that were wrapped mid-sentence are rejoined. Resumes are split into sections by recognizing common headers ("Work Experience", "TECHNICAL SKILLS", "## Projects"), bullets are tagged with their section and the job title they sit under, and total years of experience are estimated from date ranges. Job descriptions are split into *required*, *responsibility*, and *preferred* requirements, while "About us", benefits, and EEO boilerplate are dropped. Inline hints like "is a plus" demote a requirement to preferred.
2. **Skills** (`skills.py`). A curated vocabulary of about 120 skills with aliases ("k8s" is Kubernetes, "Postgres" is PostgreSQL). Short or ambiguous names such as Go, R, React, and REST are matched case-sensitively so "react quickly" or "the rest of the team" do not count.
3. **Matching** (`matcher/`). Every resume bullet and every requirement is embedded, and cosine similarity gives a requirements by bullets matrix. Each requirement's best similarity is rescaled to 0..1 and blended with how many of its named skills the resume has ("PyTorch or TensorFlow" needs only one). "N+ years" requirements also consider the resume's timeline. The overall score is a weighted average (required 1.0, responsibilities 0.8, preferred 0.5) blended with keyword coverage.
4. **Gaps and suggestions** (`suggestions/`). Requirements are labeled strong, partial, or gap. Suggestions are generated for the biggest weighted shortfalls first, as described above.
5. **Reports** (`report/`). Markdown, plain text, or JSON.

### Embedding backends

| Backend | When it is used | Notes |
| --- | --- | --- |
| `sentence-transformers` (`all-MiniLM-L6-v2`) | default when installed | understands paraphrases ("built data pipelines" and "ETL") |
| `tfidf` | automatic fallback, or `--backend tfidf` | fully offline and instant; matches on shared words, stems, and canonical skills, so it misses pure paraphrases |

The score thresholds are calibrated separately for each backend, but absolute numbers are not comparable across backends. Compare jobs using the same one.

## Design decisions (v1)

- **Heuristics over an LLM for parsing.** Header and bullet heuristics are fast, free, deterministic, and handle the large majority of resumes. `parse-resume` and `parse-job` make it easy to see when they get something wrong.
- **Everything local for suggestions.** Suggestions are rule based, which is what makes the no-fabrication guarantee checkable. An optional LLM rewriting step could be added later behind the same guard.
- **Requirements are full sentences, skills are extracted separately.** Sentence-level requirements keep context ("design reviews" versus "design"), and the skill list powers keyword coverage and the missing-keyword report.

## Limitations

- Scanned PDFs (images) need OCR first; the tool reports when a PDF has no extractable text.
- Two-column PDF layouts can interleave text; exporting the resume as single-column PDF or plain text works best.
- The skill vocabulary is tech-leaning. Unknown terms are still caught by the "other terms" check and by embeddings, and adding entries to `SKILLS` in `skills.py` is straightforward.
- Scores are a guide for prioritizing and tailoring, not a prediction of any real screening system.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run offline using the TF-IDF backend and a stand-in model for the sentence-transformers path. `scripts/make_sample_pdf.py` regenerates `data/sample_resume.pdf` from the markdown sample (needs `reportlab`).

```
data/                      sample resume (md + pdf) and two job descriptions
src/resume_matcher/
  parser/                  text extraction, resume and job description parsing
  matcher/                 embedding backends and scoring
  suggestions/             gap analysis and rewrite suggestions
  report/                  markdown, text, and JSON rendering
  analysis.py              end-to-end pipeline
  history.py               local score history
  skills.py                skill vocabulary and matching
  cli.py                   command line interface
app/streamlit_app.py       web UI
tests/                     pytest suite
```

## License

MIT
