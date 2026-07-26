# Terminology

- **Control**: original CNN → goal-conditioned GTrXL-128 → Actor/Value student.
- **Persistent**: the added state is carried across 128-step collector rollout boundaries and reset only on true done.
- **Reset128**: same module and parameters as the persistent arm, but the added state is reset every 128 steps. Reset128 is not a long-memory model; it is a matched control.
- **Memory**: internal state used during inference and online decision-making.
- **Replay**: reuse of old episodes during training.
- **Forward memory horizon**: how long forward recurrent/state information can be carried.
- **Training credit horizon**: how far gradients are actually unrolled or assigned.
- **Replay sequence horizon**: length of continuous replay sequence used in a training update.

Do not invent unverified expansions for project abbreviations.

