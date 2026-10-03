# Papers

External papers kept **next to the notes taken from them**, so any idea can be
checked against the source without a network round-trip.

Convention, matching the rest of `docs/`:

- `NNNN.NNNNNvN.IDEA.md` — the notes: what the paper claims, what it measured,
  what it admits it did not, and (for this repo) which claims are testable here.
  These are **ideas, not plans and not results**. `../../INDEX.md` and
  `../../../RESULTS.md` win over anything written here.
- `NNNN.NNNNNvN.paper.txt` — the paper's own extracted text, unedited apart from
  HTML-to-text conversion. Figures are not in the text extract; the plots carry
  the quantitative detail, so read them in the original.
- `NNNN.NNNNNvN.paper.html.gz` — the raw fetched HTML, for fidelity.

| paper | notes | why it is here |
|---|---|---|
| [arXiv:2609.12303v1](https://arxiv.org/html/2609.12303v1) — *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models* (UW + Meta FAIR) | [`2609.12303v1.IDEA.md`](2609.12303v1.IDEA.md) | External evidence for the byte-level thesis, plus two concrete untested levers for TMT (an inserted boundary symbol, and distillation — which is cheap precisely because TMT's vocabulary is already 256 bytes). Also the source of the §6.6 ruler warning: **lower BPB is not a better model**. |