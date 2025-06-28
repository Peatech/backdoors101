import math, copy, torch, numpy as np
from typing import List, Tuple
# ---------- Algorithm 1 : centred linear CKA ------------------------------
def _center(A): return A - A.mean(0, keepdims=True)
def _hsic(K,L): n = K.size(0); return torch.trace(K@L)/((n-1)**2+1e-12)
def cka(X:torch.Tensor, Y:torch.Tensor)->float:
    Xc, Yc = _center(X), _center(Y)
    K, L = Xc@Xc.T, Yc@Yc.T
    return (_hsic(K,L)/math.sqrt(_hsic(K,K)*_hsic(L,L)+1e-12)).item()
# ---------- Algorithm 2 : FedAvgCKA --------------------------------------
class FedAvgCKA:
    def __init__(self, model_template, root_loader, layer, device, drop=0.5):
        self.template, self.root_loader, self.layer, self.dev = \
            model_template, root_loader, layer, device
        self.drop=drop; self.last_sim=None
    def _acts(self, weights):
        m=copy.deepcopy(self.template).to(self.dev)
        m.load_state_dict(weights, strict=True); m.eval(); buf=[]
        h=dict(m.named_modules())[self.layer]\
              .register_forward_hook(lambda _,__,out: buf.append(out.reshape(out.size(0),-1).detach()))
        with torch.no_grad():
            for x,_ in self.root_loader: m(x.to(self.dev))
        h.remove(); return torch.cat(buf,0).cpu()
    def filter_and_aggregate(self, locals_w:List[dict]):
        acts=[self._acts(w) for w in locals_w]
        m=len(acts); sim=torch.zeros(m,m)
        for i in range(m):
            for j in range(i,m):
                s=cka(acts[i],acts[j]); sim[i,j]=sim[j,i]=s
        self.last_sim=sim
        mean=sim.mean(1); keep=(mean>=torch.quantile(mean,self.drop))
        keep_idx=keep.nonzero(as_tuple=False).flatten().tolist()
        kept=[locals_w[i] for i in keep_idx]
        agg=copy.deepcopy(kept[0])
        for k in agg.keys():
            agg[k].zero_()
            for w in kept: agg[k]+=w[k]
            agg[k]/=len(kept)
        return keep_idx, kept, agg, sim
