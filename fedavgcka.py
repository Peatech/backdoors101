import math, copy, torch
import matplotlib.pyplot as plt
import seaborn as sns

class LayerwiseCKAVisualizer:
    def __init__(self, model_template, root_loader, layer_names, device):
        self.template = model_template.to(device)
        self.root_loader = root_loader
        self.layer_names = layer_names
        self.device = device

    def _extract_activations(self, w):
        model = copy.deepcopy(self.template).to(self.device).eval()
        model.load_state_dict(w, strict=True)
        buffers, handles = {}, []
        for layer in self.layer_names:
            buffers[layer] = []
            mod = dict(model.named_modules())[layer]
            handles.append(
                mod.register_forward_hook(
                    lambda _m, _i, out, buf=buffers[layer]:
                        buf.append(out.detach().reshape(out.size(0), -1))
                )
            )
        with torch.no_grad():
            for x, _ in self.root_loader:
                model(x.to(self.device))
        for h in handles:
            h.remove()
        return {layer: torch.cat(buffers[layer], 0).cpu()
                for layer in self.layer_names}

    @staticmethod
    def _cka(X, Y):
        def center(A): return A - A.mean(0, keepdims=True)
        def hsic(K, L):
            n = K.size(0)
            return torch.trace(K @ L) / ((n - 1) ** 2 + 1e-12)
        Xc, Yc = center(X), center(Y)
        K, L = Xc @ Xc.T, Yc @ Yc.T
        return (hsic(K, L) / math.sqrt(hsic(K, K) * hsic(L, L) + 1e-12)).item()

    def compute_similarity_matrices(self, weights_list):
        m = len(weights_list)
        acts = [self._extract_activations(w) for w in weights_list]
        sims = {}
        for layer in self.layer_names:
            sim = torch.zeros(m, m)
            for i in range(m):
                for j in range(i, m):
                    s = self._cka(acts[i][layer], acts[j][layer])
                    sim[i, j] = sim[j, i] = s
            sims[layer] = sim
        return sims

    def plot_heatmaps(self, sims, client_ids=None):
        n = len(self.layer_names)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()
        for idx, layer in enumerate(self.layer_names):
            mat = sims[layer].numpy()
            mean, sd = mat.mean(), mat.std()
            vmin, vmax = mean - 3*sd, mean + 3*sd
            sns.heatmap(mat,
                        ax=axes[idx],
                        cmap='magma',
                        vmin=vmin,
                        vmax=vmax,
                        cbar_kws={'label':'CKA'},
                        square=True,                
                        annot=False,     # turn off annotations if too dense
                        xticklabels=client_ids,
                        yticklabels=client_ids
                       )
            axes[idx].set_title(f'Layer: {layer}')
            axes[idx].set_xlabel('Client idx')
            axes[idx].set_ylabel('Client idx')
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        plt.show()
        return fig
