# Tip 1 matched-capacity configs

The per-method configs one directory up were each tuned separately, so they differ
in attention heads, dropout, learning rate and EMA as well as in action
representation. Comparing them answers "which complete method wins", not "what does
the action representation change". These variants pin every shared hyperparameter to
the FAST config's values so the representation is the only factor left moving.

`prediction_horizon` is deliberately **not** matched: 59 for the tokenized arms and 60
for the continuous arms is a property of how each family frames the chunk, not a tuning
knob. Disclose it rather than forcing it.

Matched values (from `gpt_transformer.yaml`): `number_of_heads=4`,
`dropout_rate=0.4`, `attention_dropout=0.15`, `lr=1e-4`, `use_ema=true`.

Report these as the primary result and the untouched per-method configs as a tuned
secondary. If the two orderings agree the conclusion is robust; if they disagree the
gap is hyperparameter-sensitive and must be reported as such, never by picking the
favourable run.
