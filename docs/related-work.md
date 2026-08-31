# Related work and novelty boundary

The closest work is Huang et al.,
[Evidence-RL: Towards Evidence-intensive Visual Reasoning](https://arxiv.org/abs/2608.08021).
Evidence-RL uses counterfactual evidence disentanglement during GRPO to compare answer
support after interventions on target evidence regions and matched non-evidence
regions. It evaluates evidence-intensive visual reasoning across multiple benchmarks.
This artifact does not claim counterfactual grounding, anti-shortcut RL, or causal
evidence dependence as new.

Zhang et al.,
[Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](https://aclanthology.org/2026.acl-long.950/)
introduce CaRR and C-GRPO for cited text evidence chains in deep search. Zhou et al.,
[Enhancing Reinforcement Learning for Radiology Report Generation with Evidence-aware Rewards and Self-correcting Preference Learning](https://aclanthology.org/2026.acl-long.1718/)
use evidence-aware rewards that distinguish supported findings, missed findings, and
unsupported content. These establish that citation/evidence precision in RL is not a
new general idea.

Feng et al.,
[RewardMap](https://arxiv.org/abs/2510.02240), address sparse rewards in fine-grained
visual reasoning with a multi-stage curriculum and difficulty-aware rewards. Wang et
al., [MuSe](https://aclanthology.org/2026.acl-long.476/), use SFT and GRPO for
multi-stage visualized graph sampling and reasoning. MuSe is adjacent; its abstract
does not establish a recall-only versus minimal-citation reward comparison.

The narrow intended contribution is a controlled chart-QA test of:

1. explicit semantic answer-plus-provenance output;
2. overcitation induced by a matched recall-only reward;
3. minimal-set precision under multiple valid provenance sets;
4. generalization to held-out executable chart program composition.

Any claim beyond that boundary requires separate evidence.
