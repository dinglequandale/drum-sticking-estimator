---
name: project-kinematic-onset-emission
description: "Design insight for the (deferred) vision leg — score the wrist whose acceleration/rebound is phase-locked to the audio onset, not static hand position"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

When the vision leg of the fusion model ([[project-fusion-architecture]]) is eventually built, the per-hit vision emission should be a **kinematic signature matched to the audio onset**, NOT a static-position heuristic.

Specifically: the striking hand is the wrist that shows **max downward acceleration immediately followed by an instantaneous rebound** at the audio-onset timestamp. Why this is better than "which hand is lowest" (the heuristic the first probe used, which failed — see [[project-vision-probe-findings]]):
- It **sidesteps stick-decoupling**: you match the wrist's *motion phase* to the onset, so the hand needn't be near the impact point (the stick tip is ~30-40 cm away).
- The **rebound** discriminates a true strike from a feint/preparatory lift.

This converged independently from a Gemini-probed schema and from our own reasoning; Konrad and I both flagged it as the game-changer for making vision usable. Bank it for the vision leg.

**Caveat (does NOT change "vision deferred"):** it doesn't beat the empirical walls the probe found — at 30 fps a fast 16th is ~2 frames, too few to resolve an accel-then-rebound curve; an occluded far hand has no kinematics to read. So this improves vision *when vision is viable* (controlled / overhead / high-fps capture), not on arbitrary found footage. Audio emission stays always-on; vision is additive and confidence-gated.
