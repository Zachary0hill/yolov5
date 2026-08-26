# Vision Camera Intelligence Enhancement Plan

## 1. Purpose

Evolve the current webcam preview into a reliable, local-first perception system that can identify multiple people,
hands, and objects; preserve their identities over time; understand who is holding what; and recognize motion-driven
events without turning uncertain predictions into confident labels.

This plan is intentionally phased. Each phase must pass its exit gate before the next phase expands the system's
responsibilities.

## 2. Current Baseline

The current implementation provides:

- YOLOE-26 nano segmentation with open-vocabulary prompts.
- Person, cell phone, remote-control, and over-ear-headphone detection.
- ByteTrack object IDs.
- MediaPipe tracking for up to two hands and 21 landmarks per hand.
- Anatomical left/right labels, raised-finger estimates, and live orientation switching.
- Fixed class colors, segmentation overlays, class thresholds, ignore zones, and multi-frame confirmation.
- Basic nearest-hand association and suppression of implausible background phone detections.
- Local MPS inference on macOS with CPU fallback.

The main limitations are detection flicker, fragile identity through occlusion, proximity-based hand/object association,
planar finger counting, no persistent hand identity, and no temporal interpretation of motion.

## 3. Product Goals

1. Keep object and hand labels stable through lighting changes, short occlusions, fast motion, and overlapping subjects.
2. Track multiple instances of the same class without frequently switching identities.
3. Associate objects with the correct hand and person, including transfers and two-handed holds.
4. Replace frame-by-frame guesses with confidence-aware states and motion events.
5. Maintain an interactive MacBook-camera preview with measurable accuracy and performance.
6. Make failures observable and correctable so captured examples can improve later model versions.

## 4. Non-Goals

- **Face recognition or identity inference:** people receive temporary session IDs only.
- **Cloud surveillance or continuous remote upload:** inference and recordings remain local by default.
- **Safety-critical decisions:** detections and events are advisory and may be uncertain.
- **Unlimited object vocabulary in the first release:** each new class must pass the scenario benchmark.
- **Training a detector from scratch:** custom training is deferred until prompt and verifier performance is measured.
- **Full-body action recognition:** initial events are limited to observable object, hand, and motion state changes.

## 5. Guiding Design Decisions

- Keep anatomical handedness separate from screen position and preview mirroring.
- Prefer an explicit `uncertain` or `unknown` state over a confident but unstable label.
- Use temporal evidence for state changes; a single frame must not trigger an event.
- Keep class thresholds, state thresholds, and release thresholds distinct.
- Preserve one stable track identity while confidence fluctuates or an object is briefly hidden.
- Extend `vision_camera.py` and `hand_track.py` until a second consumer creates real shared-code pressure.
- Add a separate headless benchmark runner because evaluation is distinct from the interactive preview.
- Keep recorded benchmark media outside Git; version its manifest, annotations, and evaluation results.

## 6. Target Perception Pipeline

```text
Camera or recorded scenario
  -> frame quality and orientation
  -> YOLOE object/person segmentation
  -> MediaPipe hand landmarks
  -> persistent object and hand tracks
  -> person/hand/object relationships
  -> motion and gesture state machines
  -> events, overlay, screenshots, and benchmark metrics
```

Every visible result should be representable as structured state:

```text
Phone #3
  class: cell phone
  owner: Person #1
  held_by: Right Hand #2
  state: moving
  direction: toward camera
  confidence: stable
```

## 7. Scenario Matrix

The benchmark must cover these scenario families:

| Family | Required scenarios |
| --- | --- |
| Image quality | Low light, RGB lighting, backlight, glare, motion blur, noisy image |
| Scale and framing | Small distant object, close object, partial edge crop, rapid scale change |
| Ambiguity | Phone versus remote, headphones versus background shapes, screens, posters, reflections |
| Occlusion | Object behind hand, hand behind object, subject behind subject, brief full disappearance |
| Multiplicity | Two identical objects, several classes, two people, two hands, more than two hands |
| Identity | Crossing paths, leaving and returning, object transfer, hands crossing, preview flip |
| Hand pose | Palm/front/back/side views, bent fingers, thumb across palm, fist, gloves, two-handed hold |
| Motion | Stationary, pickup, release, placement, transfer, approach, retreat, entry, exit, dwell |
| Camera behavior | Camera shake, camera repositioning, source change, dropped frames, temporary disconnect |
| Performance | Long session, maximum visible detections, masks on/off, MPS and CPU fallback |

## 8. Success Metrics

Phase 0 records the actual baseline. The following are initial release targets and may be tightened after baseline data
exists:

| Metric | Release target |
| --- | --- |
| Object precision at IoU 0.5 | At least 90% across the approved object set |
| Object recall at IoU 0.5 | At least 85% across the approved object set |
| Phone-versus-remote false classification | Below 5% on the ambiguity scenarios |
| Handedness accuracy | At least 98% after per-source orientation calibration |
| Exact static finger-count accuracy | At least 90% on the annotated hand-pose set |
| Hand/object association accuracy | At least 95% on held-object scenarios |
| Track identity switches | Fewer than 2 per tracked minute in the crossing/occlusion set |
| Event precision and recall | At least 90% for pickup, release, placement, and transfer |
| Event latency | Confirmed within 500 ms of annotated event time |
| Preview performance | At least 15 FPS at the agreed default settings on the target MacBook |
| Runtime stability | 30-minute soak without crash, unbounded state growth, or camera lockup |

Metrics must be reported per scenario as well as globally. A strong easy-scene result must not hide a failing low-light
or occlusion scenario.

## 9. Phase Plan

### Phase 0 — Benchmark, Instrumentation, and Calibration

**Objective:** Establish reproducible evidence before changing detection behavior.

**Build outline:**

- Define a manifest for recorded clips, frame ranges, expected entities, relationships, events, and scenario tags.
- Add a headless benchmark command that uses the same inference path as the preview and writes JSON results.
- Record representative clips for every required scenario family using the target camera and room.
- Annotate a small high-value set first: phone/remote ambiguity, two hands, crossed hands, occlusion, and fast movement.
- Measure object precision/recall, false positives, ID switches, handedness, finger counts, associations, event timing,
  frame time, and memory use.
- Record camera source, preview orientation, resolution, model, device, thresholds, and application commit with results.
- Add a short startup calibration that verifies source orientation and can persist the selected mirror mode.

**Exit gate:**

- [ ] The same clip produces deterministic evaluation output within documented tolerances.
- [ ] Baseline metrics exist for every current capability.
- [ ] At least one failure clip exists for each planned Phase 1–5 enhancement.
- [ ] The live preview and benchmark share the same detection and hand-processing behavior.
- [ ] Benchmark media is ignored by Git while manifests and result summaries are versioned.

### Phase 1 — Temporal Tolerance and Detection Lifecycle

**Objective:** Prevent flicker and preserve believable state through noisy frames and brief detection loss.

**Build outline:**

- Replace the current hit-only confirmation with explicit candidate, confirmed, occluded, and expired states.
- Add per-class confirmation, release, and maximum-missed-frame thresholds.
- Smooth confidence, boxes, and masks across a short window without adding visible lag.
- Retain confirmed tracks through brief occlusion and reconnect them using class, trajectory, overlap, and appearance.
- Distinguish a low-confidence tracked object from a new detection.
- Detect poor image quality and report `low light` or `motion blur` rather than silently lowering every threshold.
- Bound and expire all temporal state so long-running sessions do not grow indefinitely.

**Exit gate:**

- [ ] No one-frame object labels appear under default settings.
- [ ] A confirmed object survives the approved short-occlusion duration without receiving a new ID.
- [ ] False positives decrease without more than a 5-point recall regression from Phase 0.
- [ ] State and memory remain bounded during a 30-minute soak.
- [ ] Preview performance remains at or above the release target.

### Phase 2 — Multi-Object Classification and Identity

**Objective:** Reliably track multiple similar and ambiguous objects in the same scene.

**Build outline:**

- Centralize supported-class prompt, display name, color, thresholds, aliases, and context rules in one registry.
- Preserve distinct IDs for multiple instances of the same class through crossing paths and partial occlusion.
- Add a cropped second-pass verifier for ambiguous phone, remote, and headphone detections.
- Use segmentation shape, object scale, and temporal evidence as verifier inputs where useful.
- Add stationary-background memory so persistent phone-like objects do not repeatedly become new foreground objects.
- Permit an `unknown object` result when detection evidence is strong but class evidence is weak.
- Make maximum detections and supported prompts configurable without editing inference logic.

**Exit gate:**

- [ ] Two identical objects keep separate IDs in crossing and occlusion clips.
- [ ] Phone-versus-remote error meets the release target.
- [ ] Background props and objects displayed on screens remain suppressed in the benchmark.
- [ ] Unsupported or ambiguous objects are not forced into an approved class.
- [ ] Adding a class requires registry data and benchmark evidence, not new conditional branches throughout the loop.

### Phase 3 — Multi-Hand Identity and Finger Confidence

**Objective:** Track several hands consistently and report finger states only when geometry supports them.

**Build outline:**

- Make maximum hand count configurable in the unified camera.
- Assign persistent hand track IDs using landmarks, trajectory, handedness evidence, and short occlusion memory.
- Smooth handedness over time and keep it anatomical regardless of preview orientation.
- Preserve hand identity when hands cross, rotate, or leave and re-enter briefly.
- Replace binary planar rules with 3D joint geometry and per-finger confidence.
- Add orientation-aware evaluation for palm, back-of-hand, side, foreshortened, and partially hidden poses.
- Display uncertain fingers differently and exclude them from a confident exact count.

**Exit gate:**

- [ ] Mirroring changes display orientation without corrupting anatomical handedness.
- [ ] Two crossing hands retain their IDs and labels within the target switch rate.
- [ ] Static exact-count and per-finger targets pass across all annotated orientations.
- [ ] Partial or ambiguous fingers display uncertainty instead of a confident wrong count.
- [ ] More than two hands are handled up to the configured device-performance limit.

### Phase 4 — Person, Hand, and Object Relationships

**Objective:** Determine who owns each hand and which hand is actually holding each object.

**Build outline:**

- Replace expanded-box proximity with landmark-to-mask contact and distance scoring.
- Assign hands to people using wrist/arm position, person masks, pose context, and temporal continuity.
- Support one hand holding multiple objects and one object held by two hands.
- Maintain relationship confidence separately from object-class confidence.
- Add relationship states: near, touching, held, two-handed, transferred, and released.
- Require temporal confirmation before changing ownership.
- Render concise labels such as `Phone #3 · Right Hand #2 · Person #1` only when the relationship is stable.

**Exit gate:**

- [ ] Held-object association meets the release target.
- [ ] Nearby background objects are not marked held without contact evidence.
- [ ] Two-handed holds and multiple objects in one hand are represented without discarding relationships.
- [ ] An object transfer changes ownership once and preserves the object track ID.
- [ ] Relationship labels remain stable through short hand/object occlusion.

### Phase 5 — Motion Intelligence and Event State Machines

**Objective:** Explain what tracked entities are doing over time.

**Build outline:**

- Calculate smoothed position, velocity, direction, scale change, and stationary duration per track.
- Estimate and subtract global camera motion before classifying object motion.
- Add normalized zones for entry, exit, dwell, placement, and restricted-area events.
- Implement explicit state machines for pickup, held, release, placement, transfer, approach, and retreat.
- Require minimum duration and evidence before emitting an event; add cooldowns to prevent duplicate events.
- Store a bounded in-memory event log with timestamp, involved track IDs, confidence, and optional screenshot.
- Add a debug overlay for trajectories and states that can be hidden in the normal view.

**Exit gate:**

- [ ] Pickup, release, placement, and transfer meet event precision, recall, and latency targets.
- [ ] Camera shake does not trigger bulk object-motion events.
- [ ] One physical action produces one event, not repeated notifications.
- [ ] Event records reference stable object, hand, and person IDs.
- [ ] Zone behavior remains correct at different camera resolutions.

### Phase 6 — Temporal Gestures and Presentation

**Objective:** Recognize intentional gestures while keeping the normal preview readable.

**Build outline:**

- Start with a constrained gesture set: open palm, fist, point, thumbs-up, pinch, wave, and horizontal swipe.
- Recognize dynamic gestures from tracked landmark sequences rather than isolated frames.
- Add gesture start, active, completed, cancelled, and cooldown states.
- Prevent object holding from accidentally triggering incompatible gestures.
- Separate clean, diagnostic, and presentation overlay modes.
- Add collision-aware label placement, compact labels, and optional confidence/track details.
- Keep keyboard controls and status discoverable without permanently covering the frame.

**Exit gate:**

- [ ] Each approved gesture passes positive, negative, mirrored, and object-in-hand clips.
- [ ] A gesture fires once per deliberate performance and respects its cooldown.
- [ ] Labels do not obscure the tracked hand or object when free display space is available.
- [ ] Clean mode contains only actionable labels; diagnostic details remain one toggle away.

### Phase 7 — Correction Capture and Custom Model Improvement

**Objective:** Turn observed mistakes into a controlled improvement loop for the user's actual objects and environment.

**Build outline:**

- Add controls to mark the selected detection correct, incorrect, or relabeled.
- Save the original frame, crop, model prediction, confidence, track context, orientation, and correction locally.
- De-duplicate nearly identical examples and cap storage through an explicit retention setting.
- Review class balance, lighting, pose, scale, and negative examples before training.
- Compare prompt-only YOLOE, second-stage verification, and fine-tuned model candidates on the frozen benchmark.
- Version model files, prompt registries, dataset manifests, thresholds, and evaluation summaries together.
- Promote a model only if it improves target scenarios without exceeding agreed performance regressions.

**Exit gate:**

- [ ] Corrections are reproducible, reviewable, and never silently treated as training truth.
- [ ] Dataset splits prevent adjacent frames from leaking across train and evaluation sets.
- [ ] A candidate beats the current model on the frozen benchmark and passes live validation.
- [ ] Reverting to the previous model and configuration requires only a documented option change.

### Phase 8 — Hardening and Release

**Objective:** Make the enhanced system predictable for regular use rather than only controlled demonstrations.

**Build outline:**

- Profile capture, hand inference, object inference, tracking, rendering, and event processing independently.
- Add adaptive quality presets that change documented resolution/model settings without changing semantics.
- Handle camera permission failure, source loss, model failure, unsupported MPS operations, and CPU fallback clearly.
- Verify all toggles, orientation behavior, screenshots, ignore zones, and headless evaluation paths.
- Run the complete frozen benchmark, long soak, and live acceptance checklist on the target MacBook.
- Document installation, one-command startup, controls, model provenance, benchmark results, and known limitations.
- Keep event capture and screenshots opt-in, local, and visibly indicated.

**Exit gate:**

- [ ] All release metrics pass on a clean environment.
- [ ] The 30-minute soak and camera reconnect checks pass.
- [ ] No regression exists in orientation, handedness, masks, screenshots, or keyboard controls.
- [ ] Model and dependency licenses are documented.
- [ ] The release has a documented rollback command and known-limitations section.

## 10. Phase Dependencies

| Phase | Depends on | Enables |
| --- | --- | --- |
| 0. Benchmark | Current implementation | Evidence-based work and regression gates |
| 1. Tolerance | Phase 0 | Stable downstream identity and events |
| 2. Multi-object | Phases 0–1 | Reliable object identity and ambiguity handling |
| 3. Multi-hand | Phases 0–1 | Reliable hand identity and finger state |
| 4. Relationships | Phases 2–3 | Ownership, holding, and transfer understanding |
| 5. Motion events | Phases 1 and 4 | Stateful scene interpretation |
| 6. Gestures | Phases 3 and 5 | Intentional temporal controls |
| 7. Learning loop | Phase 0 and observed failures | Environment-specific accuracy improvements |
| 8. Hardening | All shipping phases | Repeatable local release |

Phases 2 and 3 may proceed independently after Phase 1. Phase 4 must not begin until both produce stable identities.

## 11. Verification Strategy

Each phase uses the narrowest meaningful validation first, followed by the shared scenario benchmark:

1. Geometry and state-machine checks with synthetic landmarks and tracks.
2. Deterministic runs against annotated recorded clips.
3. Negative clips designed to provoke false positives.
4. Live camera checks for usability and latency.
5. Performance profiling on MPS and documented CPU fallback.
6. Full frozen-benchmark comparison against the prior accepted commit.

A phase cannot be accepted from screenshots alone. Its benchmark report must include configuration, commit, aggregate
metrics, per-scenario failures, performance, and intentional target changes.

## 12. Release Gates

- **Accuracy gate:** All applicable metrics pass globally and by required scenario family.
- **Regression gate:** No previously accepted scenario falls beyond its documented tolerance.
- **Identity gate:** Track and relationship IDs remain stable through approved crossing and occlusion clips.
- **Performance gate:** Target FPS and bounded memory pass on the target MacBook.
- **Failure gate:** Uncertainty, source loss, and model failure are visible and recoverable.
- **Privacy gate:** Recording and correction capture are opt-in and stored locally by default.
- **Documentation gate:** Startup, controls, configurations, limitations, and rollback are current.

## 13. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Open-vocabulary prompts remain weak for exact headphones or remotes | Add a verifier first; fine-tune only after benchmark evidence |
| ByteTrack loses identity when detection disappears | Add bounded occlusion state and appearance/trajectory reconnection |
| MediaPipe handedness changes with mirroring or camera source | Calibrate per source and test anatomical labels independently of display position |
| 3D landmarks remain unreliable under occlusion | Report per-finger uncertainty and avoid a forced exact count |
| Two inference stages reduce FPS | Run the verifier only on ambiguous confirmed crops and profile each stage |
| Camera movement looks like object movement | Estimate global frame motion before entity motion |
| Event rules become scattered conditionals | Centralize state transitions once motion events become a shared responsibility |
| Saved correction images expose private surroundings | Make capture explicit, local, visible, reviewable, and easy to delete |
| Benchmark improves while live behavior regresses | Require both frozen-clip metrics and live acceptance checks |

## 14. Open Decisions

These do not block Phase 0 but must be resolved before the named phase:

- **Product:** Which additional object classes are worth supporting in Phase 2?
- **Product:** Should `unknown object` be visible in normal mode or diagnostic mode only?
- **Engineering:** What short-occlusion duration feels stable without incorrectly reconnecting new objects?
- **Engineering:** What maximum hand count preserves the target FPS on the target MacBook?
- **Product:** Which gestures should cause actions versus only display labels in Phase 6?
- **Privacy:** How long should correction crops and event screenshots be retained?
- **Release:** Is the first supported environment this room/camera or varied portable environments?

## 15. Recommended First Implementation Slice

Begin with Phase 0 and the smallest useful part of Phase 1:

1. Record and annotate five short clips: clean baseline, phone/remote ambiguity, crossed hands, held-object occlusion, and
   fast pickup/release.
2. Add the headless JSON benchmark path.
3. Capture current accuracy, identity, finger, association, event, FPS, and memory baselines.
4. Replace hit-only confirmation with confirmed/occluded/expired lifecycle states.
5. Re-run the five clips and accept the slice only if flicker and identity improve without violating performance or
   recall tolerances.

This slice creates the evidence and temporal foundation needed by every later phase while keeping the first change
small enough to diagnose and reverse.
