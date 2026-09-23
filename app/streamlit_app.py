"""Streamlit front end: the project's findings for a non technical reader.

Reads only app/assets/ (built by build_app_assets.py). No Manifesto text
is shipped or shown; the try it box classifies whatever the visitor types.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# validated palette slots (light, dark); diverging poles plus neutral midpoint
PALETTE = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "over": "#2a78d6", "under": "#e34948", "mid": "#c3c2b7"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "over": "#3987e5", "under": "#e66767", "mid": "#383835"},
}

st.set_page_config(page_title="Can you trust the percentages?", page_icon=":bar_chart:", layout="centered")


@st.cache_data
def load_assets():
    summary = json.loads((ASSETS / "summary.json").read_text(encoding="utf-8"))
    names = json.loads((ASSETS / "category_names.json").read_text(encoding="utf-8"))
    return summary, names


@st.cache_resource(show_spinner="Loading the language model (first visit only)...")
def load_model():
    from sentence_transformers import SentenceTransformer
    z = np.load(ASSETS / "model.npz", allow_pickle=False)
    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return encoder, {k: z[k] for k in z.files}


def colors():
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        mode = "light"
    return PALETTE[mode]


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


summary, names = load_assets()
c = colors()
ppi5 = summary["ppi"]["0.05"]

st.title("Can you trust the percentages?")
st.write(
    "Researchers often turn large amounts of text into percentages: how much of a party manifesto "
    "is about the economy, or how many survey answers mention price. Doing that by hand is slow, so "
    "the obvious move is to let a model do the coding. This page shows what happens to the "
    "percentages when you do, and a cheap way to fix them."
)
st.caption(
    f"Tested on {summary['test']['manifestos']} English election manifestos "
    f"({summary['test']['sentences']:,} sentences, {summary['test']['categories']} policy categories) "
    "from parties the model never saw during training."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sentences coded correctly", pct(summary["item"]["accuracy"]))
k2.metric("Content counted in the wrong category", pct(ppi5["model_only"], 1))
k3.metric("After checking 5% by hand", pct(ppi5["with_checks"], 1))
k4.metric("Sentences a person still needs to see", pct(summary["review"]["review_load"]))

# ---------------------------------------------------------------- try it
st.header("Try it")
st.write(
    "Type a sentence the way it might appear in a manifesto. The model suggests the policy "
    "category and says whether it is sure enough to code it on its own."
)
text = st.text_area("Your sentence", value="Every child deserves a free, high quality school place.",
                    label_visibility="collapsed")
if st.button("Classify", type="primary") and text.strip():
    encoder, m = load_model()
    x = encoder.encode([text.strip()], normalize_embeddings=True)
    z = (x - m["mean"]) / m["scale"]
    logits = z @ m["coef"].T + m["intercept"]
    p = np.exp(logits - logits.max())
    p = (p / p.sum()).ravel()
    order = np.argsort(-p)
    top = pd.DataFrame({"Category": [names.get(m["classes"][i], m["classes"][i]) for i in order[:3]],
                        "Confidence": p[order[:3]]})
    plausible = int((p >= 1 - float(m["lac_threshold"])).sum())

    bars = alt.Chart(top).mark_bar(cornerRadiusEnd=4, color=c["s1"], size=18).encode(
        x=alt.X("Confidence:Q", axis=alt.Axis(format="%", grid=True), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Category:N", sort=None, title=None),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Confidence:Q", format=".0%")],
    ).properties(height=130)
    st.altair_chart(bars, use_container_width=True)

    if plausible == 1:
        st.success(f"Auto coded as **{top['Category'][0]}**. Only one category is plausible at the 90% level.",
                   icon=":material/check_circle:")
    else:
        st.warning(f"Send to a person. {plausible} categories are still plausible at the 90% level, "
                   "so the model should not decide this one alone.", icon=":material/person_search:")
    st.caption("This box uses the smaller of the two models in the project, so its guesses are rough. "
               "That is part of the point.")

# ---------------------------------------------------------------- the drift
st.header("Why the percentages drift")
st.write(
    "A model that gets half the sentences right sounds like it should give roughly the right "
    "percentages, because the mistakes might cancel out. They do not. The model leans towards the "
    "categories it saw most often in training, so common topics get counted too often and rare ones "
    "too rarely. More training data shrinks the problem, but it levels off well above zero."
)

curve = pd.DataFrame(summary["learning_curve"])
long = curve.melt(id_vars="train_sentences", value_vars=["count_top_guess", "average_probabilities"],
                  var_name="method", value_name="wrong")
long["method"] = long["method"].map({"count_top_guess": "Count the top guess",
                                     "average_probabilities": "Average the probabilities"})
domain = ["Count the top guess", "Average the probabilities"]
color = alt.Color("method:N", scale=alt.Scale(domain=domain, range=[c["s1"], c["s2"]]),
                  legend=alt.Legend(title=None, orient="top"))
base = alt.Chart(long).encode(
    x=alt.X("train_sentences:Q", title="Sentences used for training", scale=alt.Scale(type="log"),
            axis=alt.Axis(values=list(curve["train_sentences"]), format=",")),
    y=alt.Y("wrong:Q", title="Content counted in the wrong category", axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 0.22])),
    color=color,
)
lines = base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True, stroke=None).encode(
    tooltip=[alt.Tooltip("method:N", title="Method"),
             alt.Tooltip("train_sentences:Q", title="Training sentences", format=","),
             alt.Tooltip("wrong:Q", title="Wrong category", format=".1%")])
ends = base.transform_filter(alt.datum.train_sentences == int(curve["train_sentences"].max())).mark_text(
    align="left", dx=8, fontSize=12).encode(text=alt.Text("wrong:Q", format=".1%"))
st.altair_chart((lines + ends).properties(height=320), use_container_width=True)
st.caption("Counting each sentence's top guess is what most people do. Averaging the model's "
           "probabilities instead already removes about a third of the error, at no extra cost.")
with st.expander("Show the numbers"):
    st.dataframe(curve.rename(columns={"train_sentences": "Training sentences",
                                       "count_top_guess": "Wrong, top guess",
                                       "average_probabilities": "Wrong, averaged",
                                       "accuracy_proxy_alpha": "Agreement with humans (alpha)"}),
                 hide_index=True)

st.subheader("Which topics get over or under counted")
bias = pd.DataFrame(summary["category_bias"][:12])
bias["direction"] = np.where(bias["bias_points"] > 0, "Counted too often", "Counted too rarely")
bias["label"] = bias["name"].str.slice(0, 48)
bars = alt.Chart(bias).mark_bar(cornerRadiusEnd=4, size=14).encode(
    x=alt.X("bias_points:Q", title="Percentage points off, average per manifesto"),
    y=alt.Y("label:N", sort=alt.EncodingSortField("bias_points", order="descending"), title=None),
    color=alt.Color("direction:N", scale=alt.Scale(domain=["Counted too often", "Counted too rarely"],
                                                   range=[c["over"], c["under"]]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=[alt.Tooltip("name:N", title="Category"),
             alt.Tooltip("bias_points:Q", title="Points off", format="+.1f"),
             alt.Tooltip("true_share:Q", title="True share (%)", format=".1f")],
)
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=c["mid"]).encode(x="x:Q")
st.altair_chart((bars + zero).properties(height=380), use_container_width=True)
st.caption("The twelve categories with the largest error. Welfare, market regulation and infrastructure "
           "are inflated. Political corruption is the biggest undercount: it was rare in the training parties "
           "and common in the test parties.")
with st.expander("Show the numbers"):
    st.dataframe(bias[["name", "bias_points", "true_share"]].rename(
        columns={"name": "Category", "bias_points": "Points off", "true_share": "True share (%)"}),
        hide_index=True)

# ---------------------------------------------------------------- the fix
st.header("The fix: check a small sample by hand")
st.write(
    "Instead of trusting the model's percentages, a person codes a small random sample of each "
    "manifesto. The gap between the model and the person on that sample tells you how far off the "
    "model is, and you correct the full count by that amount. The method is called prediction "
    "powered inference, and it comes with an error margin you can actually trust."
)
budget = st.select_slider("Share of each manifesto checked by hand", options=["5%", "10%", "20%"], value="5%")
row = summary["ppi"][str(int(budget[:-1]) / 100)]
f1, f2, f3 = st.columns(3)
f1.metric("Model only", pct(row["model_only"], 1), help="Content counted in the wrong category")
f2.metric(f"Model plus {budget} checked", pct(row["with_checks"], 1),
          delta=f"{100 * (row['with_checks'] - row['model_only']):.1f} points", delta_color="inverse")
f3.metric("Error margins that hold", pct(row["interval_hit_rate"]),
          help="How often the 95% margin contains the true share, for topics above 2% of a manifesto")
st.caption(
    "With this small model, checking a sample by hand removes almost all of the bias. It does not yet "
    "save effort compared with just coding the sample and ignoring the model, because the model's "
    "guesses are too rough to add much. A stronger model is being trained to test exactly that."
)

st.header("How sure is the model, sentence by sentence?")
st.write(
    f"The model can also say when it is unsure and hand those sentences to a person. Set to be right "
    f"{pct(summary['review']['target'])} of the time, it was right {pct(summary['review']['coverage'])} "
    f"of the time, but it passed {pct(summary['review']['review_load'])} of sentences to a person. "
    f"The few it coded alone were right {pct(summary['review']['auto_accuracy'])} of the time. "
    "A small model is honest about being unsure; it is just unsure about almost everything."
)

st.divider()
st.caption(
    "Data: Manifesto Project (WZB), corpus version 2026-1, coded with the May 2021 coding handbook. "
    "No manifesto text is shown on this page. Code and full method: see the GitHub repository."
)
