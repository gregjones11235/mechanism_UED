# RMT16 Phase4A V2.2 — Publication Status Errata

**Errata issued in:** V2.3 (`RMT16_PHASE4A_V2_3_FORMAL_IDENTITY_AND_CERTIFICATE_FINALIZATION`)
**Branch:** `henry/rmt16-phase4a-v2-original-vtrace`
**Affected round:** V2.2 (`RMT16_PHASE4A_V2_2_RUNTIME_BINDING_AND_PROTOCOL_COMPLETENESS`)
**Director residual item:** 8 (V2.2 report says NOT_PUSHED but `f2b7aead` IS now remote HEAD)

## The discrepancy

The V2.2 label file `reports/rmt16_phase4a_v2_2_labels.json` records, accurately **at the time
V2.2 was finalized**:

```
"V2_2_REMOTE_PUBLICATION_STATUS": "NOT_PUSHED",
"V2_2_PUSH_PERFORMED": false,
```

After V2.2 finalization, the V2.2 commit was pushed. The remote HEAD of
`henry/rmt16-phase4a-v2-original-vtrace` is now:

```
f2b7aead44426825f905fa8b82c5f66c29ee167a   (the V2.2 commit; parent 87d1e552...)
```

So the V2.2 publication-status labels are historically correct for the moment they were
written, but they no longer describe the current remote state: **V2.2 is PUSHED**.

## Correction policy (why the V2.2 files are NOT rewritten)

Per standing evidence discipline, finalized round reports are immutable evidence artifacts.
The V2.2 files are therefore **NOT** edited:

- `reports/rmt16_phase4a_v2_2_labels.json` — unchanged (GATE38 still asserts its original
  `V2_2_REMOTE_PUBLICATION_STATUS=NOT_PUSHED` content, and GATE50 asserts the file was not
  rewritten).
- `reports/rmt16_phase4a_v2_2_final.md` — unchanged.

The correction is carried **forward**, in V2.3, by this errata and by the V2.3 labels:

```
"V2_2_ERRATUM_REMOTE_PUBLICATION_STATUS": "PUSHED",
"V2_2_ERRATUM_REMOTE_HEAD": "f2b7aead44426825f905fa8b82c5f66c29ee167a",
"V2_3_PUBLICATION_STATUS_AT_COMMIT_CREATION": "NOT_PUSHED",
"V2_3_PUSH_PERFORMED_BEFORE_COMMIT": false,
```

The V2.3 keys are deliberately **creation-time** labels: they state that no push had
happened when the V2.3 commit was created, and make NO claim about the current remote
state — so a later push by 总控 cannot contradict them (the exact failure mode this
errata documents for V2.2).

## Current layered publication truth (as of V2.3 finalization)

| Layer | Status | Commit |
| --- | --- | --- |
| Base (pre-V2.2) | PUSHED (PASS) | `87d1e552415d292417dcb6e6f9f6b16b97a6d135` |
| V2.2 | PUSHED (per this errata) | `f2b7aead44426825f905fa8b82c5f66c29ee167a` |
| V2.3 | NOT_PUSHED **at V2.3 commit creation** (creation-time fact; no current-remote claim) | local commit on this branch at creation time |

No unscoped `PUSH_PERFORMED` key is introduced (the §九 layering rule remains in force).
The V2.3 round performed **no push before its commit creation**; whether 总控 later pushes
V2.3 is deliberately NOT asserted here — asserting it would re-create exactly the failure
mode this errata records for V2.2.
