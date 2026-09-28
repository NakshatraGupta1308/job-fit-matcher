"""The "Tailor your resume" panel shown under each job's report in the Streamlit app."""

from __future__ import annotations

import streamlit as st

from resume_matcher.analysis import Analysis
from resume_matcher.matcher.embeddings import Embedder
from resume_matcher.tailoring import (
    PROJECTS_ANCHOR,
    Addition,
    GapAnswer,
    TailoringPlan,
    anchor_options,
    build_gap_prompts,
    bullet_index,
    cover_letter_line,
    draft_bullet,
    learning_plan,
    notes_markdown,
    pdf_available,
    rescore,
    to_markdown,
    to_pdf,
    to_text,
)

NOT_ANSWERED, YES, NO = "Not answered", "Yes, I have", "No, not yet"


def render_tailor_panel(a: Analysis, embedder: Embedder, key: str) -> None:
    st.divider()
    st.subheader("Tailor your resume for this job")
    st.caption(
        "Accept the edits you agree with and answer the gap questions honestly. A new bullet is only drafted from what "
        "you type, and gaps you have not covered yet get a cover letter line and a learning plan instead."
    )
    score_slot = st.container()

    resume = a.resume
    plan = TailoringPlan()
    notes = []

    rewrites = [s for s in a.suggestions if s.has_rewrite]
    st.markdown("##### 1. Edits to existing bullets")
    if not rewrites:
        st.write("No wording edits suggested for this job.")
    for i, s in enumerate(rewrites):
        with st.container(border=True):
            st.markdown(f"**For:** {s.requirement or 'General polish'}")
            st.markdown(f"Current: {s.original}")
            apply = st.checkbox("Apply this edit", key=f"{key}_edit_{i}")
            text = st.text_input("Edited bullet", value=s.suggested, key=f"{key}_edit_text_{i}", disabled=not apply)
            if s.caution:
                st.caption(s.caution)
            idx = bullet_index(resume, s.original)
            if apply and idx is not None and text.strip():
                plan.edits[idx] = text.strip()

    prompts = build_gap_prompts(a)
    anchors = anchor_options(resume)
    anchor_keys = [k for k, _ in anchors]
    st.markdown("##### 2. Fill the gaps")
    if not prompts:
        st.write("No gaps that a new bullet could cover. Nice.")
    for i, p in enumerate(prompts):
        with st.expander(f"{p.requirement}", expanded=i == 0):
            st.write(p.question)
            choice = st.radio("Your answer", [NOT_ANSWERED, YES, NO], horizontal=True, key=f"{key}_gap_{i}", label_visibility="collapsed")
            if choice == YES:
                answer = GapAnswer(True)
                answer.what = st.text_input(
                    "What did you do?", key=f"{key}_what_{i}", placeholder="built a pipeline that streams order events to the reporting service"
                )
                c1, c2 = st.columns(2)
                answer.tools = c1.text_input("Tools or technologies", value=", ".join(p.tools), key=f"{key}_tools_{i}")
                answer.result = c2.text_input(
                    "Result or scale (a number helps)", key=f"{key}_result_{i}", placeholder="processing 2M events per day"
                )
                default = anchor_keys.index(p.suggested_anchor) if p.suggested_anchor in anchor_keys else len(anchors) - 1
                where = st.selectbox(
                    "Where does it go?", range(len(anchors)), index=default, format_func=lambda j: anchors[j][1], key=f"{key}_where_{i}"
                )
                answer.anchor = anchor_keys[where]
                if not answer.what.strip():
                    st.info("Describe what you did and a bullet will be drafted here.")
                    continue
                draft = draft_bullet(answer, a.job, resume)
                st.markdown("**Draft bullet**")
                st.code(draft.text, language=None, wrap_lines=True)
                for tip in draft.tips:
                    st.caption(f"Tip: {tip}")
                override = st.text_input("Want to reword it? (optional)", key=f"{key}_override_{i}")
                text = override.strip() or draft.text
                if draft.problems and not override.strip():
                    st.error(" ".join(draft.problems))
                    continue
                if st.checkbox("Add this bullet to my resume", value=True, key=f"{key}_add_{i}"):
                    plan.additions.append(Addition(answer.anchor or PROJECTS_ANCHOR, text))
                    new = [t for t in p.tools if t.lower() in text.lower()]
                    if new and st.checkbox(f"Also add {', '.join(new)} to my Skills section", value=True, key=f"{key}_skills_{i}"):
                        plan.new_skills += [t for t in new if t not in plan.new_skills]
            elif choice == NO:
                line = cover_letter_line(p, resume)
                plans = learning_plan(p)
                notes.append((p, line, plans))
                st.markdown("Nothing is added to your resume. Here is an honest way to handle it instead.")
                st.markdown("**Cover letter line** (edit before using)")
                st.code(line, language=None, wrap_lines=True)
                for lp in plans:
                    st.markdown(f"**How to build real experience with {lp.skill}**")
                    st.markdown("\n".join(f"{n}. {step}" for n, step in enumerate(lp.steps, 1)))

    st.markdown("##### 3. Your tailored resume")
    after = a if plan.is_empty else rescore(resume, plan, a.job, embedder)
    with score_slot:
        c1, c2, c3 = st.columns(3)
        c1.metric("Score before", f"{a.score:.0f}")
        c2.metric("Score after your changes", f"{after.score:.0f}", delta=f"{after.score - a.score:+.0f}" if not plan.is_empty else None)
        c3.metric("Changes", plan.change_count())

    if plan.is_empty:
        st.write("Apply an edit or add a bullet above to build a tailored version.")
    else:
        with st.expander("Preview", expanded=False):
            st.markdown(to_markdown(resume, plan))
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Resume (.md)", to_markdown(resume, plan), file_name="tailored_resume.md", key=f"{key}_dl_md", disabled=plan.is_empty)
    d2.download_button("Resume (.txt)", to_text(resume, plan), file_name="tailored_resume.txt", key=f"{key}_dl_txt", disabled=plan.is_empty)
    if pdf_available():
        d3.download_button(
            "Resume (.pdf)", to_pdf(resume, plan) if not plan.is_empty else b"", file_name="tailored_resume.pdf",
            mime="application/pdf", key=f"{key}_dl_pdf", disabled=plan.is_empty,
        )
    d4.download_button(
        "Gap notes (.md)", notes_markdown(a.job_name, notes), file_name="gap_notes.md", key=f"{key}_dl_notes", disabled=not notes
    )
