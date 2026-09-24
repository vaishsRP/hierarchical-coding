"""Streamlit front end: the project's findings for a non technical reader.
Answer first, then one tab per dataset, then what it means.

Reads only app/assets/ (built by build_app_assets.py). No Manifesto text or
survey answers are shipped or shown; the try it box classifies whatever the
visitor types.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# validated categorical slots 1 to 4 (light, dark)
PALETTE = {
    "light": {"experts": "#2a78d6", "model": "#eb6834", "rule": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"experts": "#3987e5", "model": "#d95926", "rule": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}

st.set_page_config(page_title="Can you trust the percentages?", layout="centered")


@st.cache_data
def load_assets(version):
    """`version` is the files' modification times, so a redeploy with new
    assets is not served from the old cache."""
    summary = json.loads((ASSETS / "summary.json").read_text(encoding="utf-8"))
    names = json.loads((ASSETS / "category_names.json").read_text(encoding="utf-8"))
    return summary, names


@st.cache_resource(show_spinner="Loading the language model, first visit only")
def load_model():
    from sentence_transformers import SentenceTransformer
    z = np.load(ASSETS / "model.npz", allow_pickle=False)
    return SentenceTransformer(EMBED_MODEL, device="cpu"), {k: z[k] for k in z.files}


def colors():
    try:
        return PALETTE["dark" if st.context.theme.type == "dark" else "light"]
    except AttributeError:
        return PALETTE["light"]


def number_card(label, value, note=""):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"<div style='font-size:1.9rem;font-weight:600;line-height:1.15'>{value}</div>",
                    unsafe_allow_html=True)
        if note:
            st.caption(note)


summary, names = load_assets(tuple(f.stat().st_mtime for f in sorted(ASSETS.glob("*.json"))))
c = colors()
ppi = summary["ppi"]
wrong = ppi["0.05"]["model_only"]
fixed = ppi["0.05"]["with_checks"]
big = summary["big_models"]
bes = summary["bes"]

# ------------------------------------------------------------------ the answer first
st.title("Can you trust the percentages?")
st.markdown(
    "When a computer sorts text into topics, what people report is a percentage: *how much of this "
    "manifesto is about the economy*, *how many survey answers mention prices*. "
    "I tested whether those percentages can be trusted."
)
with st.container(border=True):
    st.markdown("**What I found**")
    st.markdown(
        f"1. **Not on their own.** A model that tags sentences about as well as a trained human still files "
        f"{wrong:.0%} of a manifesto under the wrong topic, and a bigger model does worse.\n"
        f"2. **A small hand check fixes it.** Checking 5% of each manifesto brings the error to {fixed:.1%}.\n"
        f"3. **Short survey answers are easier.** When a major survey switched from human to AI coding, "
        f"its numbers barely moved."
    )

manifestos, survey, try_it = st.tabs(["Manifestos", "Survey answers", "Try it yourself"])

# ------------------------------------------------------------------ manifestos
with manifestos:
    st.markdown(
        f"Experts tagged {summary['test']['sentences']:,} sentences from {summary['test']['manifestos']} "
        f"election manifestos with one of {summary['test']['categories']} policy topics. A model trained on "
        f"other parties' manifestos tagged the same sentences."
    )

    st.subheader("1. The model's percentages drift")
    st.markdown("Topics with the biggest errors: the share of an average manifesto, by the experts and by the model.")
    top = pd.DataFrame(summary["category_bias"][:8])
    top["model_share"] = top["true_share"] + top["bias_points"]
    long = pd.concat([
        pd.DataFrame({"Topic": top["name"], "Who": "Experts", "Share": top["true_share"] / 100}),
        pd.DataFrame({"Topic": top["name"], "Who": "Model", "Share": top["model_share"] / 100}),
    ])
    order = list(top.sort_values("true_share", ascending=False)["name"])
    y = alt.Y("Topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13))
    rule = alt.Chart(top).transform_calculate(
        lo="datum.true_share / 100", hi="datum.model_share / 100", Topic="datum.name").mark_rule(
        color=c["rule"], strokeWidth=2).encode(y=y, x="lo:Q", x2="hi:Q")
    dots = alt.Chart(long).mark_circle(size=140, opacity=1).encode(
        y=y,
        x=alt.X("Share:Q", title="Share of an average manifesto", axis=alt.Axis(format="%", grid=True)),
        color=alt.Color("Who:N", scale=alt.Scale(domain=["Experts", "Model"], range=[c["experts"], c["model"]]),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=13)),
        tooltip=[alt.Tooltip("Topic:N"), alt.Tooltip("Who:N"), alt.Tooltip("Share:Q", format=".1%")],
    )
    st.altair_chart((rule + dots).properties(height=340), use_container_width=True)
    st.markdown(
        "Mistakes do not cancel out. The model **inflates common topics** and **shrinks rare ones**, "
        f"so across all topics {wrong:.1%} of an average manifesto ends up in the wrong place."
    )

    st.subheader("2. A better model makes it worse")
    a, b = st.columns(2)
    with a:
        number_card("Small model", f"{wrong:.1%}", "filed under the wrong topic")
    with b:
        number_card("Bigger, fine tuned model", f"{big['flat']['cc_total_bias']:.1%}", "filed under the wrong topic")
    st.markdown(
        "The bigger model tags slightly more sentences correctly, but it gets there by betting on common "
        "topics, which inflates them further. It never picks 17 of the 63 topics at all."
    )

    st.subheader("3. Topic right, side wrong")
    st.markdown("The small model usually finds the topic, then picks whichever side it saw more often in training.")
    flips = pd.DataFrame(summary["stance_flips"])
    st.dataframe(pd.DataFrame({
        "Sentences that are": flips["stance"],
        "Tagged correctly": (100 * flips["correct"]).round().astype(int).astype(str) + "%",
        "Tagged as the opposite side": (100 * flips["flipped"]).round().astype(int).astype(str) + "%",
    }), hide_index=True, use_container_width=True)
    st.caption(
        f"An AI chatbot model gets the side right far more often, but misfiles "
        f"{summary['llm_wrong_topic']:.0%} of the content overall, the worst of all."
    )

    st.subheader("4. The fix: a person checks a small sample")
    st.markdown(
        "A person tags a random sample of each manifesto. The gap between model and person on that sample "
        "shows how far off the model is, and the full count is corrected by that amount."
    )
    pick = st.segmented_control("How much does a person check?", ["5%", "10%", "20%"], default="5%") or "5%"
    row = ppi[str(int(pick[:-1]) / 100)]
    a, b = st.columns(2)
    with a:
        number_card("Model alone", f"{row['model_only']:.1%}", "filed under the wrong topic")
    with b:
        number_card(f"With {pick} checked", f"{row['with_checks']:.1%}", "filed under the wrong topic")

# ------------------------------------------------------------------ survey answers
with survey:
    st.markdown(
        "The British Election Study asks about 30,000 people each wave: *what is the most important issue "
        "facing the country?* People coded the answers until 2023. Since 2024 an AI does. "
        "Did the numbers change when the coder did?"
    )
    series = pd.DataFrame(bes["series"])
    series["date"] = pd.to_datetime(series["date"])
    issues = ["Europe", "Immigration", "Economy", "Health"]
    issue_scale = alt.Scale(domain=issues, range=c["series"])
    base = alt.Chart(series).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=10)),
        y=alt.Y("share:Q", title="Share of answers", axis=alt.Axis(format="%")),
        color=alt.Color("issue:N", scale=issue_scale, legend=alt.Legend(title=None, orient="top", labelFontSize=13)))
    last = series[series["date"] == series["date"].max()]
    ends = alt.Chart(last).mark_text(align="left", dx=6, fontSize=12).encode(
        x="date:T", y="share:Q", text="issue:N", color=alt.Color("issue:N", scale=issue_scale, legend=None))
    switch = alt.Chart(pd.DataFrame({"date": [pd.to_datetime(bes["switch_date"])], "label": ["AI coding starts"]}))
    chart = (base.mark_line(strokeWidth=2)
             + base.mark_circle(size=40, opacity=1).encode(tooltip=[
                 alt.Tooltip("issue:N", title="Issue"), alt.Tooltip("date:T", title="Wave", format="%b %Y"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")])
             + switch.mark_rule(color=c["rule"], strokeWidth=2).encode(x="date:T")
             + switch.mark_text(align="left", dx=4, dy=-120, fontSize=12, color="gray").encode(x="date:T", text="label:N")
             + ends)
    st.altair_chart(chart.properties(height=320), use_container_width=True)
    st.caption("Brexit dominated 2016 to 2019. Shares among a random 30% of respondents.")

    st.markdown(
        "To separate the coder from the news, I trained a model on the human coded years and ran it on every "
        "wave. It acts as a fixed coder: if the AI changed how answers were sorted, the gap between the survey's "
        "numbers and this fixed coder should jump in 2024."
    )
    a, b = st.columns(2)
    with a:
        number_card("Shift at the switch", f"{bes['shift']:.1%}", "of answers moved topic")
    with b:
        number_card("Normal wobble", f"{bes['drift_median']:.1%}",
                    f"same test at fake switch points before 2024, at most {bes['drift_max']:.1%}")
    st.markdown(
        "The shift is a little larger than the normal wobble, but 2024 was an election year with new issues, "
        "so part of it may be the news rather than the coder. Either way, **the AI coder did not change the "
        "picture much.** One or two word answers like *immigration* are easy to sort."
    )

# ------------------------------------------------------------------ try it
with try_it:
    st.markdown("Write a sentence a party might put in its manifesto. The small model tags it.")
    text = st.text_area("Your sentence", value="Every child deserves a free, high quality school place.",
                        label_visibility="collapsed")
    if st.button("Tag it", type="primary") and text.strip():
        encoder, m = load_model()
        x = encoder.encode([text.strip()], normalize_embeddings=True)
        logits = ((x - m["mean"]) / m["scale"]) @ m["coef"].T + m["intercept"]
        p = np.exp(logits - logits.max())
        p = (p / p.sum()).ravel()
        order = np.argsort(-p)
        plausible = int((p >= 1 - float(m["lac_threshold"])).sum())
        guesses = pd.DataFrame({"Topic": [names.get(m["classes"][i], m["classes"][i]) for i in order[:3]],
                                "Confidence": p[order[:3]]})
        st.altair_chart(alt.Chart(guesses).mark_bar(color=c["experts"], size=22, cornerRadiusEnd=4).encode(
            x=alt.X("Confidence:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
            y=alt.Y("Topic:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
            tooltip=[alt.Tooltip("Topic:N"), alt.Tooltip("Confidence:Q", format=".0%")],
        ).properties(height=140), use_container_width=True)
        if plausible == 1:
            st.markdown(f"**Verdict: the model can tag this alone** as {guesses['Topic'][0]}.")
        else:
            st.markdown(f"**Verdict: a person should check this.** {plausible} topics are still plausible. "
                        f"The model says this about {summary['review']['review_load']:.0%} of real sentences.")
        st.caption("It spots the topic well but often gets the side wrong.")

# ------------------------------------------------------------------ conclusion
st.divider()
st.subheader("What this means")
st.markdown(
    f"Automated coding is about as accurate as a human coder, and that is exactly the problem: humans agree "
    f"with expert coding only about half the time beyond chance too. Neither gives trustworthy percentages "
    f"on its own. The fix is cheap: have a person check a small random sample and correct the totals. "
    f"That turns a {wrong:.0%} error into under 1%, with error margins you can report."
)
st.caption(
    "Data: Manifesto Project (WZB), 2021 coding handbook; British Election Study internet panel "
    "(Fieldhouse et al., 2026). No manifesto text or survey answers are shown. "
    "Code and full method on GitHub: vaishsRP/hierarchical-coding."
)
