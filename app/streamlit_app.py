"""Streamlit front end. Run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume_matcher import history as history_store  # noqa: E402
from resume_matcher.analysis import analyze  # noqa: E402
from resume_matcher.matcher.embeddings import get_embedder  # noqa: E402
from resume_matcher.matcher.scorer import GAP  # noqa: E402
from resume_matcher.parser.job_parser import parse_job_text  # noqa: E402
from resume_matcher.parser.resume_parser import parse_resume, parse_resume_text  # noqa: E402
from resume_matcher.report.report_builder import to_json, to_markdown, to_text  # noqa: E402

st.set_page_config(page_title="Resume Matcher", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(backend: str):
    return get_embedder(backend)


def read_resume(upload, pasted: str):
    if upload is not None:
        suffix = Path(upload.name).suffix.lower() or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = Path(tmp.name)
        try:
            resume = parse_resume(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        resume.source = upload.name
        return resume
    if pasted.strip():
        return parse_resume_text(pasted)
    return None


st.title("Resume to job matcher")
st.caption(
    "Scores how well your resume fits a job description and suggests edits that only rephrase what is already "
    "in your resume. Everything runs locally."
)

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Embedding backend",
        ["auto", "sentence-transformers", "tfidf"],
        help="auto uses sentence-transformers when available and falls back to offline TF-IDF.",
    )
    max_suggestions = st.slider("Max suggestions per job", 0, 20, 8)
    save = st.checkbox("Save scores to local history", value=False)

left, right = st.columns(2)
with left:
    st.subheader("Resume")
    upload = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    pasted_resume = st.text_area("...or paste resume text", height=220)
with right:
    st.subheader("Job descriptions")
    n_jobs = st.number_input("How many postings?", min_value=1, max_value=10, value=1, step=1)
    job_texts = []
    for i in range(int(n_jobs)):
        job_texts.append(st.text_area(f"Job description {i + 1}", height=220 if n_jobs == 1 else 140, key=f"job_{i}"))

if st.button("Analyze", type="primary"):
    resume = read_resume(upload, pasted_resume)
    jobs = [parse_job_text(t) for t in job_texts if t.strip()]
    if resume is None:
        st.error("Add a resume first.")
        st.stop()
    if not jobs:
        st.error("Paste at least one job description.")
        st.stop()

    embedder = load_embedder(backend)
    analyses = [analyze(resume, job, embedder=embedder, max_suggestions=max_suggestions) for job in jobs]
    if save:
        for a in analyses:
            history_store.record(a)
    st.session_state["analyses"] = analyses

analyses = st.session_state.get("analyses")
if analyses:
    st.caption(f"Backend: {analyses[0].result.backend}")
    if len(analyses) > 1:
        st.subheader("Ranking")
        ranked = sorted(analyses, key=lambda a: -a.score)
        st.table(
            [
                {"Job": a.job_name, "Score": round(a.score), "Fit": a.result.fit_label, "Gaps": len(a.gaps.gaps)}
                for a in ranked
            ]
        )

    tabs = st.tabs([f"{a.job_name[:30]} ({a.score:.0f})" for a in analyses])
    for tab, a in zip(tabs, analyses):
        with tab:
            r, g = a.result, a.gaps
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall match", f"{r.overall_score:.0f}/100")
            c1.caption(r.fit_label)
            c2.metric("Requirement coverage", f"{r.requirement_score:.0f}/100")
            c3.metric("Keyword coverage", "n/a" if r.keyword_score is None else f"{r.keyword_score:.0f}/100")

            st.markdown("#### Requirements")
            st.dataframe(
                [
                    {
                        "Requirement": m.requirement.text,
                        "Type": m.requirement.category,
                        "Score": round(m.score * 100),
                        "Status": m.label,
                        "Closest resume line": m.best_bullet.text if m.best_bullet else "",
                    }
                    for m in r.requirement_matches
                ],
                width="stretch",
                hide_index=True,
            )

            if g.missing_required_skills or g.missing_preferred_skills or g.other_missing_terms:
                st.markdown("#### Missing keywords")
                if g.missing_required_skills:
                    st.markdown("**Required:** " + ", ".join(g.missing_required_skills))
                if g.missing_preferred_skills:
                    st.markdown("**Nice to have:** " + ", ".join(g.missing_preferred_skills))
                if g.other_missing_terms:
                    st.markdown("**Other terms:** " + ", ".join(g.other_missing_terms))
                if g.listed_not_shown:
                    st.markdown("**Listed in Skills but never shown in a bullet:** " + ", ".join(g.listed_not_shown))

            st.markdown("#### Suggested edits")
            st.caption("Keep an edit only if it is true. Suggestions never add experience you have not described.")
            for s in a.suggestions:
                title = s.requirement or "General polish"
                with st.expander(title, expanded=s.has_rewrite):
                    if s.original:
                        st.markdown(f"**Current:** {s.original}")
                    if s.has_rewrite:
                        st.markdown("**Suggested:**")
                        st.code(s.suggested, language=None)
                    st.markdown(s.rationale)
                    if s.caution:
                        st.warning(s.caution)

            gaps = [m for m in r.requirement_matches if m.label == GAP]
            if gaps:
                st.markdown("#### Biggest gaps")
                for m in sorted(gaps, key=lambda m: m.score)[:5]:
                    st.markdown(f"- {m.requirement.text}")

            d1, d2, d3 = st.columns(3)
            d1.download_button("Download markdown", to_markdown(a), file_name="match_report.md", key=f"md_{id(a)}")
            d2.download_button("Download text", to_text(a), file_name="match_report.txt", key=f"txt_{id(a)}")
            d3.download_button("Download JSON", to_json(a), file_name="match_report.json", key=f"json_{id(a)}")

with st.expander("Score history"):
    entries = history_store.load()
    if entries:
        st.dataframe(
            [
                {"Date": e.get("timestamp", "")[:16].replace("T", " "), "Job": e.get("job"), "Score": e.get("score"), "Fit": e.get("fit")}
                for e in reversed(entries)
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("No saved scores yet. Tick 'Save scores to local history' before analyzing.")
