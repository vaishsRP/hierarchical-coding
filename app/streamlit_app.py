"""Streamlit front end: the project's findings as a short story for a non
technical reader.

Reads only app/assets/ (built by build_app_assets.py). No Manifesto text is
shipped or shown; the try it box classifies whatever the visitor types.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# validated categorical slots 1 and 2 (light, dark)
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


def big_number(label, value, note=""):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"<div style='font-size:2.4rem;font-weight:600;line-height:1.1'>{value}</div>",
                    unsafe_allow_html=True)
        if note:
            st.caption(note)


summary, names = load_assets(tuple(f.stat().st_mtime for f in sorted(ASSETS.glob("*.json"))))
c = colors()
ppi = summary["ppi"]
wrong_small = ppi["0.05"]["model_only"]

# ------------------------------------------------------------------ opening
st.title("Can you trust the percentages?")
st.markdown(
    f"Election manifestos are split into sentences, and experts tag each sentence with a policy topic. "
    f"The result everyone reports is a percentage: **how much of a manifesto is about welfare, the economy, "
    f"immigration.** I trained a model to do the tagging on {summary['test']['sentences']:,} sentences from "
    f"{summary['test']['manifestos']} manifestos it had never seen. It tags about "
    f"**{summary['item']['accuracy']:.0%} of sentences correctly.** So are its percentages right?"
)

# ------------------------------------------------------------------ the key chart
st.header("Not quite. Here is where they go wrong.")
st.markdown("For each topic, the share of an average manifesto according to the experts and according to the model.")

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
st.altair_chart((rule + dots).properties(height=360), use_container_width=True)

w = top.iloc[0]
cor = next(b for b in summary["category_bias"] if b["code"] == "304")
st.markdown(
    f"The model **inflates topics that are common** in its training data and **shrinks rare ones.** "
    f"{w['name']} is really {w['true_share']:.1f}% of a manifesto; the model says {w['model_share']:.1f}%. "
    f"Political corruption is really {cor['true_share']:.1f}%; the model says "
    f"{cor['true_share'] + cor['bias_points']:.1f}%."
)
with st.expander("Show the numbers"):
    st.dataframe(top[["name", "true_share", "model_share"]].rename(columns={
        "name": "Topic", "true_share": "Experts (%)", "model_share": "Model (%)"}).round(1), hide_index=True)

# ------------------------------------------------------------------ size of the problem
st.header("Across all 63 topics, it adds up")
a, b = st.columns(2)
with a:
    big_number("Content filed under the wrong topic", f"{wrong_small:.1%}",
               "In an average manifesto.")
with b:
    big_number("With a much bigger model", "16.5%",
               "It tags slightly more sentences correctly, but leans even harder on common topics.")
st.markdown(
    "Mistakes do not cancel out because they all point the same way. "
    "More training data shrinks the error, then it levels off."
)
curve = pd.DataFrame(summary["learning_curve"])
line = alt.Chart(curve).encode(
    x=alt.X("train_sentences:Q", title="Sentences the model learned from",
            scale=alt.Scale(type="log", domain=[3800, 98000], nice=False),
            axis=alt.Axis(values=list(curve["train_sentences"]), format=",")),
    y=alt.Y("count_top_guess:Q", title="Filed under the wrong topic", axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 0.22])),
    tooltip=[alt.Tooltip("train_sentences:Q", title="Sentences", format=","),
             alt.Tooltip("count_top_guess:Q", title="Wrong topic", format=".1%")],
)
st.altair_chart((line.mark_line(color=c["model"], strokeWidth=2)
                 + line.mark_circle(color=c["model"], size=80, opacity=1)).properties(height=240),
                use_container_width=True)

# ------------------------------------------------------------------ stance
st.header("Topic right, side wrong")
st.markdown("The model usually finds the topic, then picks whichever side it saw more often in training.")
flips = pd.DataFrame(summary["stance_flips"])
st.dataframe(pd.DataFrame({
    "Sentences that are": flips["stance"],
    "Tagged correctly": (100 * flips["correct"]).round().astype(int).astype(str) + "%",
    "Tagged as the opposite side": (100 * flips["flipped"]).round().astype(int).astype(str) + "%",
}), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ the fix
st.header("The fix: let a person check a small sample")
st.markdown(
    "A person tags a small random sample of each manifesto. Comparing their tags with the model's on "
    "that sample shows how far off the model is, and the full count is corrected by that amount."
)
pick = st.segmented_control("How much of each manifesto does a person check?", ["5%", "10%", "20%"],
                            default="5%") or "5%"
row = ppi[str(int(pick[:-1]) / 100)]
a, b = st.columns(2)
with a:
    big_number("Model on its own", f"{row['model_only']:.1%}", "Filed under the wrong topic")
with b:
    big_number(f"Model plus {pick} checked", f"{row['with_checks']:.1%}", "Filed under the wrong topic")
st.caption(
    "The honest catch: at this accuracy, the person's sample alone would do almost as well. "
    "The model mainly adds error margins you can trust."
)

# ------------------------------------------------------------------ survey answers
bes = summary["bes"]
st.header("What about survey answers?")
st.markdown(
    f"The British Election Study asks about 30,000 people each wave: *what is the most important issue "
    f"facing the country?* People coded the answers until 2023. Since 2024 an AI does. "
    f"Did the numbers jump when the coder changed?"
)
series = pd.DataFrame(bes["series"])
series["date"] = pd.to_datetime(series["date"])
issues = ["Europe", "Immigration", "Economy", "Health"]
issue_scale = alt.Scale(domain=issues, range=c["series"])
col = alt.Color("issue:N", scale=issue_scale, legend=alt.Legend(title=None, orient="top", labelFontSize=13))
base = alt.Chart(series).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=10)),
    y=alt.Y("share:Q", title="Share of answers", axis=alt.Axis(format="%")), color=col)
last = series[series["date"] == series["date"].max()]
ends = alt.Chart(last).mark_text(align="left", dx=6, fontSize=12).encode(
    x="date:T", y="share:Q", text="issue:N", color=alt.Color("issue:N", scale=issue_scale, legend=None))
switch = alt.Chart(pd.DataFrame({"date": [pd.to_datetime(bes["switch_date"])], "label": ["AI coding starts"]}))
lines = (base.mark_line(strokeWidth=2)
         + base.mark_circle(size=40, opacity=1).encode(tooltip=[
             alt.Tooltip("issue:N", title="Issue"), alt.Tooltip("date:T", title="Wave", format="%b %Y"),
             alt.Tooltip("share:Q", title="Share", format=".1%")])
         + switch.mark_rule(color=c["rule"], strokeWidth=2).encode(x="date:T")
         + switch.mark_text(align="left", dx=4, dy=-120, fontSize=12, color="gray").encode(x="date:T", text="label:N")
         + ends)
st.altair_chart(lines.properties(height=320), use_container_width=True)
st.caption("Shares among a random 30% of respondents, held out from training. Brexit dominated 2016 to 2019.")
a, b = st.columns(2)
with a:
    big_number("Answers that changed topic at the switch", f"{bes['shift']:.1%}")
with b:
    big_number("Normal change between years", f"{bes['drift_median']:.1%}",
               f"At most {bes['drift_max']:.1%} in 25 human coded waves.")
st.markdown(
    f"A small but real jump. On one or two word answers, AI coding stays close to human coding. "
    f"On long manifesto sentences an AI coder put {summary['llm_wrong_topic']:.0%} of content under the wrong topic."
)
with st.expander("Show the numbers"):
    st.dataframe(series.pivot_table(index="date", columns="issue", values="share").mul(100).round(1)
                 .rename_axis(None, axis=1).reset_index().assign(date=lambda d: d["date"].dt.strftime("%b %Y")),
                 hide_index=True)

# ------------------------------------------------------------------ try it
st.header("Try it yourself")
st.markdown("Write a sentence a party might put in its manifesto.")
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
    st.caption("It spots the topic well but often gets the side wrong, as shown above.")
    if plausible == 1:
        st.markdown(f"**Verdict: the model can tag this alone** as {guesses['Topic'][0]}.")
    else:
        st.markdown(f"**Verdict: a person should check this.** {plausible} topics are still plausible. "
                    f"The model says this about {summary['review']['review_load']:.0%} of real sentences.")

st.divider()
st.caption(
    "Data: Manifesto Project (WZB), coded with the 2021 handbook, and the British Election Study internet "
    "panel (Fieldhouse et al., 2026). No manifesto text or survey answers are shown here. "
    "The page uses the smaller of the two models. Code and method on GitHub: vaishsRP/hierarchical-coding."
)
