import math, copy, torch, numpy as np
import random
from torch.utils.data import DataLoader, Subset

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
    def __init__(self,
                 model_template: torch.nn.Module,
                 root_loader: DataLoader = None,
                 root_dataset: torch.utils.data.Dataset = None,
                 ref_size: int = 32,
                 layer: str = 'layer4',
                 device: torch.device = torch.device('cpu'),
                 discard_ratio: float = 0.5):
        """
        You must pass EITHER root_loader OR root_dataset.
        If root_dataset is provided, we build a <ref_size>-example loader.
        """
        self.template   = model_template
        self.layer = layer
        self.device     = device
        self.drop       = discard_ratio

        if root_loader is not None:
            self.root_loader = root_loader
        elif root_dataset is not None:
            from torch.utils.data import DataLoader, Subset
            import random
            idxs = random.sample(range(len(root_dataset)), ref_size)
            self.root_loader = DataLoader(
                Subset(root_dataset, idxs),
                batch_size=ref_size,
                shuffle=False
            )
        else:
            raise ValueError("Provide either root_loader or root_dataset")
    def _acts(self, weights):
        m=copy.deepcopy(self.template).to(self.device)
        m.load_state_dict(weights, strict=True); m.eval(); buf=[]
        h=dict(m.named_modules())[self.layer]\
              .register_forward_hook(lambda _,__,out: buf.append(out.reshape(out.size(0),-1).detach()))
        with torch.no_grad():
            for x,_ in self.root_loader: m(x.to(self.device))
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
