import json, sys, os
sys.path.insert(0, r"C:\Users\Administrator\XinSpectAI\rag")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from bm25 import BM25Index
HERE = os.path.dirname(os.path.abspath("."))
ext = BM25Index.load(r"C:\Users\Administrator\XinSpectAI\rag\index\external")
prod = BM25Index.load(r"C:\Users\Administrator\XinSpectAI\rag\index\product")

def measure(idx, q):
    hits = idx.search(q, top_k=5, prefer_lang=None)
    if not hits: return (0.0, 0.0)
    return (max(idx.answerability(q, h) for h in hits[:3]), hits[0]["score"])

def run(idx, qfile, gate, label):
    qs = json.load(open(qfile, encoding="utf-8"))
    tp=fn=tn=fp=0
    for it in qs["in_domain"]:
        a,b = measure(idx, it["q"])
        if gate(a,b): tp+=1
        else: fn+=1
    for it in qs["out_domain"]:
        a,b = measure(idx, it["q"])
        if gate(a,b): fp+=1
        else: tn+=1
    r=tp/max(tp+fn,1); rj=tn/max(tn+fp,1); p=tp/max(tp+fp,1)
    print(f"{label:34s} recall={r:.2f} reject={rj:.2f} prec={p:.2f}")
    return rj*2+r+p, label

print("=== external: 覆蓋率 AND BM25 ===")
best=[]
for at in [0.5,0.55,0.6]:
    for bt in [15,18,20,22]:
        s,l = run(ext, "questions-external.json", lambda a,b,at=at,bt=bt: a>=at and b>=bt, f"ans>={at} & bm25>={bt}")
        best.append((s,l))
best.sort(reverse=True); print("best:", best[0][1])

print("\n=== product: 覆蓋率 AND BM25 ===")
best=[]
for at in [0.5,0.55]:
    for bt in [16,20,24]:
        s,l = run(prod, "questions.json", lambda a,b,at=at,bt=bt: a>=at and b>=bt, f"ans>={at} & bm25>={bt}")
        best.append((s,l))
best.sort(reverse=True); print("best:", best[0][1])
