# Literature verification ledger

All 56 bibliography entries below were checked on 2026-08-10 against an authoritative primary record: the journal or proceedings publisher, the official JMLR/PMLR proceedings page, or a DOI registration page resolving to the publisher. The key in each bullet matches `references/verified_references.bib`. DOI URLs are intentionally retained as durable verification targets.

## Local differential privacy and information contraction

- **OhnishiAwan2025** — Direct source for locally private randomized-experiment estimators, confidence intervals, Bayesian inference, and the causal minimax benchmark. Verification: [JMLR](https://www.jmlr.org/papers/v26/23-1401.html).
- **Warner1965** — Introduces randomized response, the canonical precursor to noninteractive local differential privacy. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.1965.10480775).
- **DworkMcSherryNissimSmith2006** — Establishes sensitivity-calibrated noise and the modern differential-privacy definition used to contrast central and local models. Verification: [Springer DOI record](https://doi.org/10.1007/11681878_14).
- **KasiviswanathanEtAl2011** — Characterizes local private learning through statistical queries and distinguishes interactive from noninteractive local protocols. Verification: [SIAM DOI record](https://doi.org/10.1137/090756090).
- **DuchiJordanWainwright2013** — Develops strong data-processing inequalities and privacy-dependent statistical minimax rates in the local model. Verification: [IEEE DOI record](https://doi.org/10.1109/FOCS.2013.53).
- **KairouzOhViswanath2016** — Characterizes extremal LDP mechanisms and optimality regimes for binary and randomized-response channels. Verification: [JMLR](https://www.jmlr.org/papers/v17/15-135.html).
- **KairouzBonawitzRamage2016** — Gives order-optimal mechanisms and estimators for discrete distributions under local privacy. Verification: [PMLR](https://proceedings.mlr.press/v48/kairouz16.html).
- **DuchiJordanWainwright2018** — Provides private Le Cam, Fano, and Assouad lower bounds together with matching locally private estimators. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2017.1389735).
- **AsoodehZhang2024** — Gives sharp contraction coefficients for KL, chi-square, Hellinger, and total-variation divergences through LDP channels. Verification: [IEEE DOI record](https://doi.org/10.1109/JSAIT.2024.3397305).
- **RohdeSteinberger2020** — Shows that total-variation moduli govern broad classes of locally private minimax risks. Verification: [Project Euclid DOI record](https://doi.org/10.1214/19-AOS1901).
- **DuchiRuan2024** — Develops instance-specific local minimax complexity and explains why Fisher information is generally the wrong LDP complexity measure. Verification: [Project Euclid DOI record](https://doi.org/10.1214/22-AOS2227).
- **DuchiRogers2019** — Establishes LDP lower bounds for arbitrary interaction and all privacy levels through communication-complexity reductions. Verification: [PMLR](https://proceedings.mlr.press/v99/duchi19a.html).
- **AcharyaCanonneTyagi2020** — Develops chi-square contraction lower bounds for inference under communication and privacy channels. Verification: [IEEE DOI record](https://doi.org/10.1109/TIT.2020.3028440).
- **BarnesHanOzgur2020** — Supplies Fisher-information-based minimax lower bounds for channel-constrained distributed inference. Verification: [JMLR](https://www.jmlr.org/papers/v21/19-737.html).
- **GaboardiRogersSheffet2019** — Derives locally private mean estimators, tests, and tight confidence intervals directly relevant to privatized outcomes. Verification: [PMLR](https://proceedings.mlr.press/v89/gaboardi19a.html).
- **Raginsky2016** — Authoritative source for strong data-processing and divergence-contraction inequalities for noisy channels. Verification: [IEEE DOI record](https://doi.org/10.1109/TIT.2016.2549542).
- **Yu1997** — Concise primary reference for Assouad, Fano, and Le Cam minimax lower-bound techniques. Verification: [Springer DOI record](https://doi.org/10.1007/978-1-4612-1880-7_29).
- **Tsybakov2009** — Standard monograph treatment of minimax risk, testing reductions, and nonparametric lower bounds. Verification: [Springer](https://link.springer.com/book/10.1007/b13794).

## Randomized experiments and finite-population inference

- **Rubin1978** — Formulates Bayesian causal inference through potential outcomes and clarifies the inferential role of random assignment. Verification: [Project Euclid DOI record](https://doi.org/10.1214/aos/1176344064).
- **HorvitzThompson1952** — Introduces inverse-probability estimation, the design-based template used for exposure-specific effects. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.1952.10483446).
- **ImbensRubin2015** — Authoritative book-length treatment of potential outcomes, assignment mechanisms, and randomized-experiment inference. Verification: [Cambridge University Press](https://doi.org/10.1017/CBO9781139025751).
- **LiDing2017** — Establishes finite-population central limit theorems used for randomization-based asymptotics and coverage arguments. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2017.1295865).
- **AronowMiddleton2013** — Gives design-unbiased ATE estimators and conservative variance estimators under known randomization designs. Verification: [De Gruyter DOI record](https://doi.org/10.1515/jci-2012-0009).

## Interference, exposure mappings, and experiment design

- **OhnishiKarmakarSabbaghi2025** — Introduces the latent Degree of Interference variable and Bayesian nonparametric inference for arbitrary unknown interference. Verification: [JMLR](https://www.jmlr.org/papers/v26/24-0119.html).
- **HalloranStruchiner1995** — Early potential-outcome formulation of direct, indirect, total, and overall effects in infectious-disease settings. Verification: [publisher DOI record](https://doi.org/10.1097/00001648-199503000-00010).
- **Sobel2006** — Demonstrates how interference invalidates conventional randomized-study contrasts and formalizes partial interference in a social experiment. Verification: [publisher DOI record](https://doi.org/10.1198/016214506000000636).
- **HudgensHalloran2008** — Defines direct, indirect, total, and overall causal effects under partial interference and gives randomization-based estimators. Verification: [publisher DOI record](https://doi.org/10.1198/016214508000000292).
- **TchetgenTchetgenVanderWeele2012** — Systematizes identification and estimation of causal effects when interference occurs within groups. Verification: [SAGE DOI record](https://doi.org/10.1177/0962280210386779).
- **AronowSamii2017** — Provides the general exposure-mapping framework, Horvitz--Thompson estimators, variance estimators, and asymptotics under interference. Verification: [Project Euclid DOI record](https://doi.org/10.1214/16-AOAS1005).
- **LiuHudgens2014** — Derives large-sample randomization inference for direct and spillover effects in two-stage randomized designs. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2013.844698).
- **LiuHudgensBeckerDreps2016** — Develops generalized and stabilized inverse-probability estimators allowing broad forms of interference. Verification: [Oxford Academic DOI record](https://doi.org/10.1093/biomet/asw047).
- **BasseFeller2018** — Analyzes estimands and design-based inference for two-stage experiments with interference. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2017.1323641).
- **BairdBohrenMcIntoshOzler2018** — Derives power and optimal randomized-saturation designs for jointly learning direct and spillover effects. Verification: [MIT Press DOI record](https://doi.org/10.1162/rest_a_00716).
- **SinclairMcConnellGreen2012** — Develops multilevel experimental designs for detecting spillovers across units and clusters. Verification: [Wiley DOI record](https://doi.org/10.1111/j.1540-5907.2012.00592.x).
- **Manski2013** — Studies identification with social interactions and formalizes treatment-response restrictions that motivate exposure mappings. Verification: [Oxford Academic DOI record](https://doi.org/10.1111/j.1368-423X.2012.00368.x).
- **UganderKarrerBackstromKleinberg2013** — Introduces graph-cluster randomization and network-exposure conditions for reducing interference bias. Verification: [ACM DOI record](https://doi.org/10.1145/2487575.2487695).
- **EcklesKarrerUgander2017** — Analyzes network experiment designs and shows how graph-aware randomization reduces interference bias. Verification: [De Gruyter DOI record](https://doi.org/10.1515/jci-2015-0021).
- **AtheyEcklesImbens2018** — Constructs exact randomization tests and p-values for hypotheses about interference on observed networks. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2016.1241178).
- **BowersFredricksonPanagopoulos2013** — Gives a general randomization-inference framework for reasoning about interference without committing to one effect model. Verification: [Oxford Academic DOI record](https://doi.org/10.1093/pan/mps038).
- **BasseAiroldi2018** — Proves limitations of design-based identification under arbitrary and network interference, motivating explicit structural assumptions. Verification: [SAGE DOI record](https://doi.org/10.1177/0081175018782569).
- **SavjeAronowHudgens2021** — Shows when standard difference-in-means remains interpretable under unknown interference and characterizes its variance. Verification: [Project Euclid DOI record](https://doi.org/10.1214/20-AOS1973).
- **Savje2024** — Separates exposure-defined estimands from exposure restrictions and analyzes robustness to misspecified exposure mappings. Verification: [Oxford Academic DOI record](https://doi.org/10.1093/biomet/asad019).
- **Leung2020** — Develops consistent inference for treatment and spillover effects from a single network under network dependence. Verification: [MIT Press DOI record](https://doi.org/10.1162/rest_a_00818).
- **Leung2022** — Relaxes exact neighborhood interference to decaying approximate interference and supplies valid large-sample inference. Verification: [Econometric Society DOI record](https://doi.org/10.3982/ECTA17841).
- **ForastiereAiroldiMealli2021** — Identifies direct and spillover effects in observational networks using neighborhood exposure and generalized propensity scores. Verification: [publisher DOI record](https://doi.org/10.1080/01621459.2020.1768100).
- **OgburnVanderWeele2014** — Extends causal diagrams to represent interference pathways and clarifies identification assumptions. Verification: [Project Euclid DOI record](https://doi.org/10.1214/14-STS501).
- **PapadogeorgouMealliZigler2019** — Defines cluster- and population-level treatment-allocation policies with interference and derives design-based estimators. Verification: [Wiley DOI record](https://doi.org/10.1111/biom.13049).
- **ZiglerPapadogeorgou2021** — Formalizes bipartite interference where treatment and outcome units differ and introduces corresponding causal estimands. Verification: [Project Euclid DOI record](https://doi.org/10.1214/19-STS749).
- **HarshawEtAl2023** — Gives unbiased estimation, variance estimation, and design results for bipartite experiments under a linear exposure-response model. Verification: [Project Euclid DOI record](https://doi.org/10.1214/23-EJS2111).
- **BhattacharyaMalinskyShpitser2020** — Combines structure learning with interference methods when the dependence network itself is unknown. Verification: [PMLR](https://proceedings.mlr.press/v115/bhattacharya20a.html).
- **BarkleyEtAl2020** — Develops policy estimands and inverse-probability estimators for observational studies with clustered interference. Verification: [Project Euclid DOI record](https://doi.org/10.1214/19-AOAS1314).
- **LiWager2022** — Establishes random-graph asymptotics for direct and indirect treatment-effect estimation under network interference. Verification: [Project Euclid DOI record](https://doi.org/10.1214/22-AOS2191).
- **ToulisKao2013** — Provides an early randomization-based estimator and test for causal peer-influence effects. Verification: [PMLR](https://proceedings.mlr.press/v28/toulis13.html).

## Bayesian and nonparametric interference inference

- **ForastiereEtAl2022** — Develops Bayesian generalized-propensity-score estimation for direct and spillover effects on networks. Verification: [JMLR](https://www.jmlr.org/papers/v23/18-711.html).
- **OhnishiSabbaghi2024** — Gives a flexible Bayesian analysis of two-stage experiments jointly handling interference, nonadherence, and missing outcomes. Verification: [Project Euclid DOI record](https://doi.org/10.1214/22-BA1347).
- **IshwaranJames2001** — Introduces blocked Gibbs sampling for stick-breaking priors, the computational foundation used by the latent-interference models. Verification: [publisher DOI record](https://doi.org/10.1198/016214501750332758).
