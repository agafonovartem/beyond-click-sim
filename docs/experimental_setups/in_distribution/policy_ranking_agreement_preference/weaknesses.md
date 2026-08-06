# Weaknesses and validity assessment: policy ranking agreement as a simulator-usefulness test

Conceptual assessment of the *experimental idea* behind
`runners/in_distribution/policy_ranking_agreement_preference/` (and its
interaction-based sibling `policy_ranking_agreement/`). This is not about the
implementation — see `issues.md` in this folder for code-level findings. The
question assessed here:

> For an LLM-based user simulator to be useful for "predicting" a recommendation
> policy's behavior, it should simulate user interactions with the policy well
> enough to keep policies distinguishable — therefore the policy performance
> ranking should be the same on real and simulated data.

## What the idea gets right

- **Rank agreement is the correct usefulness criterion.** A simulator used for
  evaluation doesn't need calibrated absolute metrics; it needs to preserve the
  *ordering* of systems. This is exactly how offline metrics are validated in
  IR (system-ranking correlation against online A/B outcomes), and it is a much
  more decision-relevant property than pointwise accuracy. Framing it as
  "good enough to keep policies distinguishable" is the right instinct.
- **It is a relaxation, not a strengthening, of the pointwise claim.** A
  simulator can be mediocre per-row and still rank policies correctly — testing
  agreement separately acknowledges that. Good experimental hygiene.
- **Real and simulated utility are computed on identical rows.** Same users,
  same slates, same target — the comparison is paired by construction, which
  removes a whole class of confounds most offline-eval papers have.
- **A trivial simulator (popularity) is included in the scorer roster.** That
  control turns out to be the most important piece of the whole design
  (see below).

## Where the idea is weak

### 1. K=6 policies is statistically tiny, and the metric is dominated by the easy separations

Kendall's tau over 6 items has very coarse granularity, and any sane
simulator — including a raw popularity count — will separate Random from
LightGCN. A high tau on this roster mostly demonstrates "the simulator notices
random recommendations are bad," which is close to vacuous. The discriminating
content of the experiment lives in the mid-tier pairs (ALS vs BPR vs LightGCN),
where:

- **The real ranking itself is uncertain.** ~1000 users x 10 binary outcomes
  per policy gives a standard error of roughly +/-0.005 on each real utility.
  If ALS and BPR differ by less than that in real preference rate, there *is no
  stable ground-truth ordering* for the simulator to agree with. "Rankings
  should be the same" is then unfalsifiable in the wrong direction — you can't
  tell "simulator is wrong" from "the two policies are indistinguishable in
  reality."

"Rankings should be the same" is also over-strict as a success criterion. The
operationally meaningful claim is: *a practitioner who picks the best policy by
simulated utility doesn't lose much real utility* (selection regret), or
*pairwise sign decisions are correct above some bar*. Perfect tau is both
unnecessary and unattainable once real utilities have noise.

### 2. The result is a property of the roster, not the simulator — and the setup is gameable by popularity

Policy quality is strongly correlated with the average popularity of
recommended items. A simulator that scores *nothing but item popularity* will
likely achieve high tau. If the popularity scorer gets tau ~ 0.9, then the
LLM's tau = 0.9 demonstrates almost nothing about user simulation. Conversely,
the *informative* version of the experiment is the hard sub-roster: drop Random
and Popularity, or add near-tied policy variants (same family, different
hyperparameters), and ask whether the simulator preserves order among policies
whose real utilities are 1-2% apart. That is a much harder, much more
meaningful test — and the one a skeptical reviewer will demand.

### 3. The claim it supports is narrower than "predicting policy behavior"

This is one-step, static, slate-level evaluation: fixed policies, fixed
histories, no feedback loop, utility = *mean* response over the slate. It can
support "the simulator is a usable offline pre-screening tool for static top-k
lists under slate-mean utility." It does **not** support:

- predicting deployed policy behavior (feedback loops, repeated exposure,
  drift);
- order-sensitive policy quality — a policy that gets the right 10 items in
  the wrong order is treated as identical to one that ranks them perfectly,
  since mean utility ignores the policy's rank. Saying something about policy
  *ranking* behavior requires an order-sensitive utility too (e.g., DCG of
  simulated scores over policy ranks vs DCG of real targets);
- ranking *novel* policy families. Agreement within this roster is
  interpolation. The actual use case — "will it correctly rank a policy I
  haven't evaluated?" — is leave-one-policy-out generalization, which the
  current design doesn't test.

### 4. The "real" utility is a proxy with a known bias

A recommended item the user never organically touched in the test window counts
as 0. So "real utility" measures *the policy's ability to anticipate what the
user would organically consume and like*, not "how the user would respond if
shown the item." This is the standard missing-not-at-random problem of offline
policy evaluation; it hits all methods equally, but it systematically penalizes
exploratory/serendipitous policies relative to popularity-following ones —
which feeds back into confound #2. It doesn't invalidate the setup, but the
paper must state this definition of "real" explicitly, because it changes what
the claim means.

## Recommended changes before trusting the setup

1. **Replace the single-tau headline with a pairwise decision framework.** 15
   policy pairs; for each, the real utility difference with a user-level paired
   bootstrap CI (paired observations are available — every user sees every
   policy's slate — so this is cheap and rigorous), and the simulated
   difference. Report sign agreement, with an explicit indifference zone where
   the real difference isn't distinguishable from noise. This honestly
   separates "simulator error" from "no ground truth."
2. **Make trivial baselines the bar, not a footnote.** Random scorer,
   popularity scorer, item-mean-rating scorer. The claim should be "the LLM
   simulator beats the strongest trivial simulator *on the hard pairs*," not
   "achieves high tau."
3. **Report roster sensitivity.** Full-roster tau *and* hard-sub-roster tau
   (no Random, no Popularity), ideally plus near-tie policy variants. If
   agreement collapses on the hard roster, that's a finding, not a failure —
   it directly supports the paper's "simulator quality is setup-specific"
   thesis.
4. **Per-seed reporting.** With 3 seeds, show the pairwise decisions per seed.
   If the *real* ranking itself flips across seeds, that bounds what any
   simulator can agree with.
5. **Scope the claim in the paper** to offline slate-level pre-screening, and
   state the organic-consumption proxy explicitly. Don't write "predict policy
   behavior" without the qualifiers.
6. **Report pointwise quality alongside agreement.** Tau without AUC/accuracy
   is hard to interpret — a reader needs to know whether high agreement came
   with a badly miscalibrated simulator (which would matter for other uses).

## Verdict

The idea is valid and follows an established validation paradigm — rank
preservation is the right question to ask of an evaluation simulator. But as
stated ("rankings should be the same") it is both too weak in what it can
detect (K=6, trivial separations, popularity-gameable) and too strict in what
it demands (perfect agreement against a noisy ground-truth ranking). Reframed
as *pairwise policy discrimination with uncertainty quantification, against
trivial-simulator baselines, on a roster that includes hard pairs*, it becomes
a genuinely strong experiment — and the negative/control results it would
produce (where agreement breaks down) are exactly the evidence the central
"simulator quality is evaluation-setup-specific" thesis needs.
