# How LLMs and their internal activity are visualized

A reference survey compiled 2026-11-15 from web searches, to guide the mapping work in `TODO.md`.

**Caveat:** the details below come mostly from search-result summaries. Most primary papers and tools were not opened, so check a tool's own docs before relying on a specific feature.

There are six standard forms. Each answers a different question, and none tries to show a whole model at once.

| # | Form | Question it answers | Main examples | Our map level |
|---|---|---|---|---|
| 1 | Architecture walkthroughs | How is the model built, and how does data flow through it? | Bycroft's LLM Visualization (3D), Transformer Explainer (2D) | Level 1 |
| 2 | Attention maps | Where does each token look? | BertViz, CircuitsVis, Dodrio, AttentionViz | Level 4 (attention around loops) |
| 3 | Layer-by-token heatmaps | How does the prediction form through the layers? | Logit lens, tuned lens, activation patching | Level 4; the grid form suits the level-2 overview |
| 4 | Neuron/feature dashboards | What does one unit respond to? | OpenAI Neuron Viewer, Anthropic dashboards / SAE Vis, Neuronpedia | Level 3 (and the level-2 lookup panel) |
| 5 | Feature atlases | How do thousands of units relate? | Google PAIR Gemma Scope explorable, Activation Atlas, Concept Explorer | Level 2 embedding map |
| 6 | Attribution graphs | How was this specific output produced? | Anthropic circuit-tracer + Neuronpedia | Level 4; the visual language suits the level-2 neighborhood graph |

---

## 1. Architecture walkthroughs
- **Brendan Bycroft's LLM Visualization** (bbycroft.net/llm, repo `bbycroft/llm-viz`): an interactive 3D walk through a GPT-style model, from token embedding through attention and the MLP to the output probabilities.
  - The demo uses nano-gpt (about 85,000 parameters) on a toy task: sorting three characters.
  - Sources disagree on whether larger GPT-2-size networks can be loaded.
- **Transformer Explainer** (Georgia Tech): a 2D browser tool that runs GPT-2 live, showing how it processes text and predicts the next token.
- Also mentioned: the PyTorch blog post "Inside the Matrix" (3D), and **AnimatedLLM** (2026, browser tool built on pre-computed traces of open LLMs).
- **Use:** teaching structure and data flow, not what particular parts mean.

## 2. Attention maps
- **BertViz** (Jesse Vig), with three views:
  - **head view:** lines linking each token to the tokens it attends to;
  - **model view:** a grid of every layer × head, for scanning patterns across the network;
  - **neuron view:** how the query and key values produce attention.
  - Supports BERT, GPT-2, BART and similar.
  - Vig observed many heads that put all their attention on the first token.
- **CircuitsVis** (TransformerLens org): the attention displays used in Anthropic-style circuits work. It works in Python/Jupyter and JavaScript/React/HTML.
- **Dodrio** and **AttentionViz**: more global views, e.g. query–key embeddings across many inputs.
- Others listed in the awesome-list: Tilde, Ecco.

## 3. Layer-by-token heatmaps
- **Logit lens:** project each layer's hidden state through the output head.
  - The grid has layers as rows and input positions as columns.
  - Each cell shows the top predicted token, colored by confidence or entropy.
  - Common implementations: Seaborn and Plotly heatmaps. **LogitLens4LLMs** produces "important layers" and "all layers" heatmaps for Llama-3.1-8B and Qwen-2.5-7B.
- **Tuned lens:** trains a small per-layer correction (an affine probe) to remove the plain logit lens's bias and drift.
- **Activation patching / causal tracing:** usually the same layer × position grid, but colored by how much each location matters causally. *These searches didn't find tooling for this; search separately if needed.*
- **LLM-Microscope:** includes a related per-token measure of how much performance drops when a layer is skipped.

## 4. Neuron and feature dashboards
- **The typical page:** top-activating text snippets with highlighted tokens, activation histograms, the tokens it promotes or suppresses, and an LLM-written explanation.
- **OpenAI Neuron Viewer:** text examples from several activation ranges (top and bottom), plus GPT-4-written explanations.
- **Anthropic feature dashboards**, reproduced as **SAE Vis** (Callum McDougall).
- **Neuronpedia:** uses SAE Vis plus a database of activating examples to host neurons and SAE features.
- **Lessons to adopt:**
  - Show top *negative* activations side by side with positive ones; a Neuronpedia dashboard for a weight-sparse transformer does this.
  - Let users test a unit on their own text. Top examples alone can mislead, which is why Neuronpedia runs live models.
  - Show the whole range of activations, not just the strongest (Distill, "Feature Visualization").

## 5. Feature atlases
- Each unit or feature becomes a point in 2D, usually via **UMAP**, clustered by similarity and labeled.
- **Google PAIR's SAE explorable** maps the 16,384 features of Gemma Scope this way.
- **Activation Atlas** (Distill; OpenAI and Google) did the same for vision models.
- **Known problem:** flat UMAP maps clutter, overlap, and distort neighborhoods. Newer work responds in two ways:
  - **Concept Explorer** (HUMAP) adds a zoomable hierarchy of landmark subsets, with UMAP at each level;
  - **Visual Exploration of Feature Relationships in SAEs** (arXiv 2511.06048) focuses on curated concept sets with topology-based (mapper graph) views.
- **For us:** make the level-2 embedding map hierarchical or zoomable, not one flat scatter plot.

## 6. Attribution graphs
- Node-and-edge diagrams for **one prompt**. They flow left to right from input tokens through internal features (and "error" nodes) to the output logits, with edge thickness showing direct effect.
- **Anthropic circuit-tracer** (open source, PyPI `circuit-tracer`) computes them.
  - **Neuronpedia** hosts an interactive frontend for exploring, annotating and intervening (changing feature values and watching the output).
  - Applied to Gemma-2-2B (GemmaScope transcoders) and Llama-3.2-1B.
  - EleutherAI has its own attribution-graph library.
- **Bottleneck:** annotating by hand takes about 2 hours per prompt for an experienced tracer. Automated approaches:
  - probe prompting (arXiv 2511.07002);
  - "LLMs Can Annotate Attribution Graphs" (arXiv 2608.02632).
- **For us:** a weight-based neighborhood graph (level 2) can borrow this visual language. Label it clearly as **static wiring**, not a traced computation for a specific prompt.

---

## Implications for the level-2 wiring map
- **Overview heatmap** (layer → layer wiring): a standard grid form, like form 3.
- **Neighborhood graph:** attribution-graph style (form 6), but built from weights, so it isn't tied to a prompt. That's less common, so label it accordingly.
- **Embedding map:** standard atlas (form 5). Make it hierarchical or zoomable to avoid UMAP clutter.
- **Lookup panel:** follow the dashboard convention (form 4): positive and negative poles, vocabulary readout, strongest connections.

## Sources
- [Anthropic: open-source circuit tracing](https://www.anthropic.com/research/open-source-circuit-tracing)
- [circuit-tracer (PyPI)](https://pypi.org/project/circuit-tracer/)
- [Neuronpedia graph info](https://neuronpedia.org/graph/info)
- [Automated Circuit Interpretation via Probe Prompting (arXiv 2511.07002)](https://arxiv.org/html/2511.07002v1)
- [LLMs Can Annotate Attribution Graphs (arXiv 2608.02632)](https://arxiv.org/pdf/2608.02632)
- [Visualizing Attention in Transformer-Based Language Representation Models (BertViz, arXiv 1904.02679)](https://arxiv.org/pdf/1904.02679)
- [A Multiscale Visualization of Attention in the Transformer Model (arXiv 1906.05714)](https://arxiv.org/pdf/1906.05714)
- [BertViz on GitHub](https://github.com/jessevig/bertviz)
- [CircuitsVis on GitHub](https://github.com/TransformerLensOrg/CircuitsVis)
- [awesome-transformer-visualization](https://awesome.ecosyste.ms/projects/github.com%2Fki-seki%2Fawesome-transformer-visualization)
- [A push towards interactive transformer decoding (LessWrong)](https://www.lesswrong.com/posts/5msxxQiTDmcDNBnkF/a-push-towards-interactive-transformer-decoding)
- [llm-viz by bbycroft](https://www.sourcepulse.org/projects/1271960)
- [Gigazine: LLM Visualization](https://gigazine.net/news/20231204-llm-visualization)
- [AnimatedLLM (arXiv 2601.04213)](https://arxiv.org/html/2601.04213v1)
- [Google PAIR: SAE explorable](https://pair.withgoogle.com/explorables/sae/)
- [Visual Exploration of Feature Relationships in SAEs with Curated Concepts (arXiv 2511.06048)](https://arxiv.org/html/2511.06048v1)
- [Navigating the Concept Space of Language Models (review)](https://www.themoonlight.io/review/navigating-the-concept-space-of-language-models)
- [SAE BeiWang 2025](https://www.sci.utah.edu/~beiwang/publications/SAE_BeiWang_2025.pdf)
- [LogitLens4LLMs (arXiv 2503.11667)](https://arxiv.org/html/2503.11667v1)
- [LogitLens from scratch with Hugging Face Transformers](https://alessiodevoto.github.io/LogitLens)
- [mlxterp logit lens tutorial](https://coairesearch.github.io/mlxterp/tutorials/01_logit_lens)
- [LLM-Microscope](https://matveymih-llm-microscope.hf.space/)
- [Announcing Neuronpedia (Alignment Forum)](https://www.alignmentforum.org/posts/BaEQoxHhWPrkinmxd/announcing-neuronpedia-platform-for-accelerating-research)
- [Neuronpedia interp explorer blog](https://www.neuronpedia.org/blog/interp-explorer)
- [arXiv 2602.23826 (describes OpenAI Neuron Viewer)](https://arxiv.org/html/2602.23826v1)
- [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [Distill: Exploring Neural Networks with Activation Atlases](https://distill.pub/activation-atlas/)
