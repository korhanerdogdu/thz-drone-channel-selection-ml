# Revision notes — TAES Correspondence

Revised manuscript: `paper_taes/thz_ml_spectrum_allocation.tex` → `thz_ml_spectrum_allocation.pdf` (15 pages).
Bibliography: `paper_taes/references.bib` (BibTeX, `IEEEtaes.bst`).
Template: `IEEEtaes.cls` / `IEEEtaes.bst`, copied unmodified from `IEEE_TAES_correspondence.zip` inside `TAES+Template.zip`, class option `correspondence`.
The superseded IEEEtran paper (`paper/`) was removed from the repository after this revision was merged into `main`; it remains in git history (commit `1286300`). Fig. 1 is now `figures/maxactive_reference_single.png`, a byte-identical copy of the notebook output.

Build: `cd paper_taes && pdflatex thz_ml_spectrum_allocation && bibtex thz_ml_spectrum_allocation && pdflatex … && pdflatex …`
Result: 0 errors, 0 overfull boxes, 0 undefined references or citations.

---

## Pass 2: final scientific, bibliographic and formatting review (October 2026)

### Changes

**Figures**
- **Figs. 2 and 3 were re-exported (presentation only).** The new files are `figures/fig_capacity_main.pdf` and `figures/fig_capacity_mobility.pdf`, both vector graphics with a PNG copy.
- How the data was obtained: a scratch **copy** of the notebook (through cell `t0comb01`, without the Section U timing benchmark) was executed in the scratchpad, never in place. It had one extra cell that saved the exact arrays the two plotting cells draw, to `figures/fig_data.npz`.
- How the data was verified: all 56 code cells printed output identical to the committed notebook, timing lines excluded. The regenerated `all_methods_comparison.png`, `ml_mobility_comparison.png` and `maxactive_reference_single.png` were **pixel-identical** to the committed files.
- How the figures were redrawn: `figures/replot_paper_figures.py` reads only that file. It uses the notebook's PCHIP log-space helper verbatim and the same colours, markers, line styles and display clip.
- What changed is layout only:
  - authored at printed size (4.9 in), with 8.5 pt labels and 8 pt ticks and legend;
  - panel tags moved to the empty upper-right corner;
  - Fig. 3(b)'s clipped x-label fixed (now "Zenith angle φ [deg]", explained in the caption);
  - legend now reads "k-NN", matching the text;
  - the x-label appears once per column in Fig. 2.
- No PNG edge carries ink. The original PNGs are unchanged, and Fig. 1 is unchanged.

**Tables**
- Table II is now full-width and grouped by model family. It also lists the D3QN details that are in the code: one replay minibatch per epoch, gradient clipping, how C_max is defined, and the output rule.
- Table IV: the note was moved below the table and the header made clearer.
- Table V: the header now groups the two misaligned columns.
- Table VI: column spacing widened.
- All numeric cells are unchanged (checked by script).

**Equations**
- All equations were checked against the code: spreading loss, Beer–Lambert law, noise, gain, SNR, capacity, EP, WF, and the double-DQN and dueling formulas. They match.
- Clarifications added:
  - the reward's actual normalization: C_max is the MaxActive-EP capacity of the context, C_min = 0, capacities in bit/s;
  - a new eq. (16) for the output rule;
  - the double-DQN target network is not used (horizon 1);
  - the azimuth condition is dropped when the boresight points to the zenith or nadir.

**Claims corrected or softened**
- Abstract: MaxActive is "best within its search family", and the D3QN result is stated as "≥ 99.7% … per-context training with reference capacity in reward".
- MaxActive optimality: the precise condition (EP, near-equal channel widths, all prefix lengths searched) under which restricting the search to prefixes loses nothing.
- **"Huber attains the best single-split accuracy" was wrong.** GaussProc is first on the single split (RMSE 0.008); Huber is second (0.183). The sentence is corrected.
- "Nearly every regressor crosses the bound" now reads "every regressor … at one or more points", which is what Table IV shows.
- "D3QN … found the same prefix" now says the chosen channel counts differ by up to 30 channels (497 vs 467 at 81 m) with near-equal capacity. Source: stored notebook output.
- "Attains 100%" now reads "99.7–100%".
- "All reported metrics use nested CV" now applies to the main dataset only; the mobility errors are labelled in-sample.
- Removed or softened:
  - "needs no floating-point accelerator";
  - "every model fits in on-chip memory";
  - "the speed-up ratio transfers" (now an unmeasured expectation);
  - "the coupled loop converges in a handful of iterations" (iteration counts were never recorded);
  - "20 epochs suffice to reach the reference" (now stated as the complete procedure including the fallback).
- Section V-B now states that training one D3QN context (0.11 s) costs more than one sweep (77 ms) and needs the reference capacity, so the D3QN branch pays off only for contexts that recur.

**References.** All entries were verified against Crossref, a publisher page or the PDF. DOIs were added in TAES template style, via `note`, because `IEEEtaes.bst` ignores the `doi` field. Completed:
- Akyildiz 2022: vol. 70, no. 6, pp. 4250–4285.
- Han 2016: pp. 910–922, from the PDF.
- Nasir 2019: vol. 37, no. 10, pp. 2239–2250.
- Xu 2021 and Clough 2005: full author lists.
- Langford 2007: corrected to the published NIPS title, "…multi-armed bandits with side information".
- Van Hasselt 2016: AAAI vol. 30, no. 1, DOI added; the unverified page range was removed.
- Wang 2016: PMLR vol. 48, pp. 1995–2003.
- The intro's altitude claim now also cites Saeed et al. (JSAC 2021).

No citation was removed. All are relevant and support the sentences that cite them.

**TAES instructions** (ieee-aess.org, "Last Updated: 18 July 2025"):
- No limit on the number of manuscript pages.
- Accepted Correspondence items pay **$200 per printed page beyond 6**.
- Two-column, single-spaced, 10 pt, PDF.
- The template zip linked from that page could not be byte-compared with the local one. The local 2022 zip is used unchanged, and the first-page empty left column is kept.

### Verification
- Numeric cells of all results and complexity tables are identical to the original manuscript.
- SHA-256 of all 71 research files (notebook, `.py`, `.m`, `.mat`, PNG, original `.tex` and PDF) is unchanged.
- The notebook was never executed in place. The scratch copy and its outputs live in the session scratchpad; only `fig_data.npz` and the re-plot script were brought into `paper_taes/figures/`.

---

## Unresolved issues requiring supervisor decision

1. **D3QN output rule (scientific).** The implementation returns the greedy action unless a better action was already evaluated during training (eq. 16). It explores 1600 actions over a 600-action space, with ε starting at 1.0. The reported 99.7–100% therefore characterizes training plus exploration plus selection, not the greedy network alone, and the notebook does not record which source produced each decision. The paper now states this openly. **A greedy-only evaluation would need a new run, which was not done.** Supervisors should decide whether that experiment is required before submission.
2. **D3QN uses the reference in training.** C_max in the reward is the MaxActive-EP capacity of the same context, inherited from Elçik et al. Training one context (0.11 s) also costs more than one MaxActive sweep (77 ms). The D3QN therefore saves computation only for recurring, pre-trained contexts; the paper now says so. This weakens the "replace the search" framing for the D3QN branch.
3. **Page charges.** At 15 pages in the Correspondence layout, publication would incur overlength charges (beyond 6 printed pages). Options: accept the charges, shorten later, or move to a Regular Paper (limit 10 printed pages before charges).
4. **AI-use disclosure (policy).** The TAES page states that AI-generated content must be disclosed in the acknowledgments, naming the system and the affected sections, and that failure to disclose can lead to rejection. A commented placeholder is in the `.tex`. **The authors must write this statement**; it was not drafted for them.
5. **IEEE membership grades** (`\member{}`) are still omitted; please supply them if any apply.
6. **Funding wording.** The TÜBİTAK sentence is the draft from the instructions; please confirm the exact wording.
7. **The feature named "log-SNR" in the code** is a received-power indicator (no noise term). The paper describes it correctly as Π_k. The code label is misleading but was not changed.
8. **MaxActive implementation granularity.** The reference sweep uses a coarse grid (step |F|/100) refined over ±100 channels, not every prefix length. For EP, the prefix optimality argument in Section II-C holds only up to this search resolution.
9. **Hu & Han (2025)** is an arXiv preprint with no journal record found. Keep it, or replace it if it has since been published.

## Pass 1 summary (kept for reference)
- Rewrote the manuscript into the six agreed sections in the official template. The abstract was written last, and Introduction, Related Work and Contributions were added.
- Moved the system model and problem formulation into the text, and removed Algorithm 1 and the old formula table.
- Added Tables I and II from the code and shrank the figures.
- Added the TÜBİTAK 224N014 funding line.
- Earlier wording corrections:
  - k-NN per-scenario sentence;
  - feature definitions (received-power indicator, log₁₀(d+1));
  - disclosure of the D3QN fallback;
  - 24 dBm taken from the MATLAB code.
