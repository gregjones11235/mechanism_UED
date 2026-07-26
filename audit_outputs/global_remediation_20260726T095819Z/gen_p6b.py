#!/usr/bin/env python
# CC4 remediation [6/6] final artifacts: registry/inventory JSON aliases, recomputation aliases, SHA256SUMS.
import json, os, csv, hashlib, shutil
BASE=os.getcwd()
OUT=open(os.path.join(BASE,"audit_outputs","_remediation_outdir.txt")).read().strip()
RPT=os.path.join(BASE,"reports","global_remediation")
def J(p,o):
    with open(p,"w",encoding="utf-8") as f: json.dump(o,f,indent=2,ensure_ascii=False)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(65536),b""): h.update(b)
    return h.hexdigest()

# evaluator_registry.json (alias of the fixed CSV as JSON)
with open(os.path.join(OUT,"global_evaluator_registry_fixed.csv"),encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
J(os.path.join(OUT,"evaluator_registry.json"),
  {"schema":"mechanism_UED.evaluator_registry/v1","canonical_anchor":"CANONICAL_EVALUATOR_V1",
   "evaluator_count":len(rows),"evaluators":rows})

# metric_recomputation.csv + claim_scope_matrix.csv audit aliases (verbatim copies)
shutil.copyfile(os.path.join(OUT,"global_metric_recomputation_fixed.csv"),os.path.join(OUT,"metric_recomputation.csv"))
shutil.copyfile(os.path.join(OUT,"global_claim_scope_matrix_fixed.csv"),os.path.join(OUT,"claim_scope_matrix.csv"))

# artifact_inventory.json over OUT + reports (exclude __pycache__)
inv=[]
for root in (OUT,RPT):
    for dp,_,fns in os.walk(root):
        if "__pycache__" in dp: continue
        for fn in sorted(fns):
            if fn=="SHA256SUMS": continue
            p=os.path.join(dp,fn); rel=os.path.relpath(p,BASE).replace("\\","/")
            inv.append({"path":rel,"sha256":sha(p),"bytes":os.path.getsize(p)})
inv.sort(key=lambda x:x["path"])
J(os.path.join(OUT,"artifact_inventory.json"),
  {"task":"GLOBAL_EVALUATION_REMEDIATION","artifact_count":len(inv),
   "outputs_root":"audit_outputs/global_remediation_20260726T095819Z",
   "reports_root":"reports/global_remediation","artifacts":inv})

# SHA256SUMS (relative to BASE) for OUT + reports
lines=[]
for a in inv:
    lines.append(f"{a['sha256']}  {a['path']}")
with open(os.path.join(OUT,"SHA256SUMS"),"w",encoding="utf-8",newline="\n") as f:
    f.write("\n".join(lines)+"\n")
print(f"artifacts inventoried: {len(inv)}")
print("WROTE evaluator_registry.json, metric_recomputation.csv, claim_scope_matrix.csv, artifact_inventory.json, SHA256SUMS")
