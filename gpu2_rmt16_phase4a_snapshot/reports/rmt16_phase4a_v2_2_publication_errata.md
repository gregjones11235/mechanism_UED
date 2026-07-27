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
"V2_3_REMOTE_PUBLICATION_STATUS": "NOT_PUSHED",
"V2_3_PUSH_PERFORMED": false,
```

## Current layered publication truth (as of V2.3 finalization)

| Layer | Status | Commit |
| --- | --- | --- |
| Base (pre-V2.2) | PUSHED (PASS) | `87d1e552415d292417dcb6e6f9f6b16b97a6d135` |
| V2.2 | PUSHED (per this errata) | `f2b7aead44426825f905fa8b82c5f66c29ee167a` |
| V2.3 | NOT_PUSHED (pending 总控复审) | local commit on this branch only |

No unscoped `PUSH_PERFORMED` key is introduced (the §九 layering rule remains in force).
The V2.3 round performs **no push**; pushing V2.3 is a later, separately-authorized step.
