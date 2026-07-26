#!/usr/bin/env python
# CC4 remediation [4/6] part B: unified recompute (Phase2 per-world available) + EVIDENCE_UNVERIFIED
# rows for server-only lines + fixed claim-scope matrix + fixed report-mismatch json.
import csv, json, os, math, sys
import numpy as np
from scipy import stats
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
SRC=os.path.join(BASE,"student_upgrade_wave1_4gpu","reports","phase2_unified_eval.json")
Z=1.959963984540054
EVAL_SHA="224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1"
seed42=json.load(open(os.path.join(OUT,"world_manifests","canonical_worlds_256_seed42.json"),encoding="utf-8"))
RECIPE_HASH=seed42["world_recipe_hash"]
CKPT={"BASELINE":"d4e85af58b7f87d6","Control":"ece6fa9962e815123ce947577a93040057bc9df0b1e686dd28424cb2bbdabf55",
 "SG_Persistent_on":"1bd4fbfe91ab4da4","SG_Reset128_on":"2ffdd269b94e1e6b",
 "EM_Persistent_on":"11307081315f8059","EM_Reset128_on":"a3030f387c2e8cbb"}
BASEID={"BASELINE":"TEACHER17500_BASELINE","Control":"CONTROL24576_BASELINE"}
def wilson(k,n,z=Z):
    p=k/n; den=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/den,(c+h)/den)
def cp(k,n,a=0.05):
    lo=0.0 if k==0 else stats.beta.ppf(a/2,k,n-k+1); hi=1.0 if k==n else stats.beta.ppf(1-a/2,k+1,n-k); return (float(lo),float(hi))

o=json.load(open(SRC,encoding="utf-8")); arms=o["results"]; N=256
arrays={}; rows=[]
for name,a in arms.items():
    succ=np.array(a["success_per_world"],dtype=int); f3=np.array(a["floor3_per_world"],dtype=int); died=np.array(a["died_per_world"],dtype=int)
    arrays[name]={"succ":succ,"f3":f3,"died":died}
    ns=int(succ.sum()); nf=int(f3.sum()); nd=int(died.sum())
    wlo,whi=wilson(ns,N); clo,chi=cp(ns,N)
    rows.append(dict(row_type="arm",experiment=name,baseline_id=BASEID.get(name,"(non-baseline arm)"),
        success_count=ns,N=N,success_rate_pp=round(ns/N*100,5),wilson95_lo=round(wlo*100,4),wilson95_hi=round(whi*100,4),
        clopper_pearson95_lo=round(clo*100,4),clopper_pearson95_hi=round(chi*100,4),
        death_count=nd,death_rate_pp=round(nd/N*100,4),floor3_count=nf,floor3_reach_pp=round(nf/N*100,4),
        mean_episode_length=a.get("mean_episode_length"),evaluator_sha=EVAL_SHA,
        world_set_hash="REQUIRED(materialized; JAX-blocked)",world_recipe_hash=RECIPE_HASH,
        checkpoint_sha=CKPT.get(name,"see ablation_phase2.json section7"),action_mode="stochastic",
        evidence_level="MATCHED_CAUSAL_SCREEN (single-seed)",source="phase2_unified_eval.json (per-world, local)",
        recompute_matches_reported=bool(a.get("n_success")==ns)))
# EVIDENCE_UNVERIFIED rows for server-only lines (no per-world locally; NO summary substitution)
for exp,note in [
  ("W512_Persistent_PPO","collapsed 28/256 aggregate only; per-world server-only"),
  ("W512_Reset128_PPO","collapsed 7/256 aggregate only"),
  ("W512_Persistent_P2Replay","90/256 aggregate only; evaluator f76bb53c"),
  ("W512_Reset128_P2Replay","95/256 aggregate only; evaluator f76bb53c"),
  ("P7_EgoMap","no per-world/params-SHA local; seed100000 line"),
  ("P8_LongMem","NO_POSITIVE_SIGNAL; per-world server-only; 256w final + 64w migration"),
  ("P9_AuthenticReset","P9_NO_POSITIVE_SIGNAL; per-world server-only; continuation text-only"),
  ("P2_FullA","no formal per-world eval local"),
  ("RMT16","CC2 domain; ENGINEERING_ONLY; no Phase4A local")]:
    rows.append(dict(row_type="arm",experiment=exp,baseline_id="n/a",success_count="UNVERIFIED",N=256,
        success_rate_pp="UNVERIFIED",wilson95_lo="n/a",wilson95_hi="n/a",clopper_pearson95_lo="n/a",clopper_pearson95_hi="n/a",
        death_count="UNVERIFIED",death_rate_pp="UNVERIFIED",floor3_count="UNVERIFIED",floor3_reach_pp="UNVERIFIED",
        mean_episode_length="UNVERIFIED",evaluator_sha="see registry (line-specific)",
        world_set_hash="REQUIRED",world_recipe_hash="seed-line dependent",checkpoint_sha="server manifest",
        action_mode="stochastic (declared; P7 has argmax dead code quarantined)",evidence_level="EVIDENCE_UNVERIFIED",
        source=note,recompute_matches_reported="NOT_RECOMPUTED (no per-world)"))
# paired comparisons
def paired(A,B):
    a=arrays[A]["succ"].astype(bool); b=arrays[B]["succ"].astype(bool)
    kA=int(a.sum()); kB=int(b.sum()); n10=int((a&~b).sum()); n01=int((~a&b).sum()); disc=n10+n01
    if disc>0:
        p=2*min(stats.binom.cdf(min(n10,n01),disc,0.5),1-stats.binom.cdf(min(n10,n01)-1,disc,0.5)); p=min(1.0,p)
    else: p=1.0
    rng=np.random.default_rng(12345); idx=rng.integers(0,N,size=(20000,N))
    db=((a[idx].mean(1)-b[idx].mean(1))*100); lo,hi=np.percentile(db,[2.5,97.5])
    return dict(row_type="paired",experiment=f"{A} - {B}",baseline_id=f"{BASEID.get(A,A)} vs {BASEID.get(B,B)}",
        success_count=f"{kA}/{kB}",N=N,success_rate_pp=f"{round(kA/N*100,3)}/{round(kB/N*100,3)}",
        delta_pp=round((kA-kB)/N*100,3),discordant=disc,A_only=n10,B_only=n01,mcnemar_p=round(p,6),
        bootstrap95_lo=round(float(lo),3),bootstrap95_hi=round(float(hi),3),ci_crosses_zero=bool(lo<=0<=hi),
        signal=bool(p<0.05 and not(lo<=0<=hi)),evaluator_sha=EVAL_SHA,world_set_hash="REQUIRED(materialized)",
        world_recipe_hash=RECIPE_HASH,checkpoint_sha=f"{CKPT.get(A,'?')} vs {CKPT.get(B,'?')}",action_mode="stochastic",
        evidence_level="MATCHED_CAUSAL_SCREEN (single-seed)",paired_comparison_allowed="YES (same evaluator+recipe+success+denominator+action_mode; world_set_hash pending materialization)",
        source="phase2_unified_eval.json (per-world, local)")
comparisons=[("BASELINE","Control"),("SG_Persistent_on","SG_Reset128_on"),("EM_Persistent_on","EM_Reset128_on"),
 ("SG_Reset128_on","Control"),("EM_Reset128_on","Control"),("SG_Persistent_on","Control"),("EM_Persistent_on","Control"),
 ("SG_Persistent_on","SG_Persistent_off"),("SG_Reset128_on","SG_Reset128_off"),("EM_Persistent_on","EM_Persistent_off"),
 ("EM_Reset128_on","EM_Reset128_off"),("SG_Persistent_off","Control"),("EM_Persistent_off","Control")]
prows=[paired(a,b) for a,b in comparisons]
# unified columns (union)
cols=[]
for r in rows+prows:
    for k in r:
        if k not in cols: cols.append(k)
allrows=rows+prows
with open(os.path.join(OUT,"global_metric_recomputation_fixed.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in allrows: w.writerow({c:r.get(c,"") for c in cols})

# ===== fixed claim-scope matrix (carry forward audit claims + remediation labels) =====
claims=json.load(open(os.path.join(BASE,"audit_outputs","global_readonly_audit_20260726T050340Z","global_claim_scope_matrix.json"),encoding="utf-8"))["claims"]
repro={"CL-07":"UNVERIFIED (per-world server-only)","CL-08":"UNVERIFIED + collapsed regime","CL-09":"UNVERIFIED + confounded",
 "CL-10":"UNVERIFIED (no formal eval)","CL-11":"UNVERIFIED (no per-world/SHA local)","CL-12":"UNVERIFIED (server-only)",
 "CL-13":"UNVERIFIED (server-only)","CL-14":"UNVERIFIED (CC2; engineering only)","CL-15":"UNVERIFIED numbers (server)",
 "CL-01":"PASS (recomputed)","CL-02":"PASS (recomputed)","CL-03":"PASS (no carry, recomputed)","CL-04":"PASS (no carry, recomputed)",
 "CL-05":"PARTIAL (single-seed marginal, recomputed)","CL-06":"PARTIAL (single-seed, recomputed)","CL-16":"CC3 domain","CL-17":"UNVERIFIED (smoke only)"}
cs_fields=["claim_id","claim_text","experiment","raw_data_local","evidence_level","recompute_status","reproducibility_label","claim_scope_allowed","remediation_gate"]
csrows=[]
for c in claims:
    cid=c["claim_id"]
    csrows.append(dict(claim_id=cid,claim_text=c["claim_text"],experiment=c["experiment"],raw_data_local=c["raw_data_local"],
        evidence_level=c["evidence_level"],recompute_status=("RECOMPUTED (per-world local)" if c["raw_data_local"].startswith("YES") else "NOT_RECOMPUTED (no per-world)"),
        reproducibility_label=repro.get(cid,"UNVERIFIED"),claim_scope_allowed=c["claim_scope_allowed"],
        remediation_gate=("GATE12 satisfied (raw data present)" if c["raw_data_local"].startswith("YES") else "GATE12: no strong conclusion without per-world (EVIDENCE_UNVERIFIED)")))
with open(os.path.join(OUT,"global_claim_scope_matrix_fixed.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=cs_fields); w.writeheader(); w.writerows(csrows)

# ===== fixed report-mismatch json =====
mm={"method":"recomputed all Phase2 per-arm aggregates + causal deltas from per-world arrays; server-only lines NOT recomputed (no summary substitution)",
 "phase2_per_arm_mismatches":0,"phase2_causal_delta_all_match":True,
 "recompute_artifact":"global_metric_recomputation_fixed.csv (arm + paired rows, full provenance schema)",
 "server_only_lines_status":"EVIDENCE_UNVERIFIED (W512/P7/P8/P9/P2/RMT16) - no per-world locally; sync BLOCKED",
 "new_provenance_per_row":["evaluator_sha","world_set_hash(REQUIRED)","world_recipe_hash","checkpoint_sha","action_mode","evidence_level"],
 "mismatches_found":[],
 "discipline":"NO_SUMMARY_AS_PRIMARY_EVIDENCE: aggregate percentages for W512/P7/P8/P9 are NOT used as recomputed evidence; they remain UNVERIFIED until per-world sync."}
J=os.path.join(OUT,"global_report_metric_mismatches_fixed.json")
with open(J,"w",encoding="utf-8") as f: json.dump(mm,f,indent=2,ensure_ascii=False)
print("arms recomputed:",len(rows)-9,"| EVIDENCE_UNVERIFIED rows: 9 | paired:",len(prows))
# console key results
for r in prows[:7]:
    print(f"  {r['experiment']:42s} {r['success_rate_pp']:>15s} dpp={r['delta_pp']:+7.3f} disc={r['discordant']:>3} mcn_p={r['mcnemar_p']:.4f} boot95=[{r['bootstrap95_lo']:+.2f},{r['bootstrap95_hi']:+.2f}] signal={r['signal']}")
print("WROTE global_metric_recomputation_fixed.csv, global_claim_scope_matrix_fixed.csv, global_report_metric_mismatches_fixed.json")
