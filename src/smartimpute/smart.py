import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .generator import Generator
from .discriminator import D1_WGAN_GP, D2_BCE
from .utils import gradient_penalty, recon_loss_fn

def train_smartimpute(
    X, M,
    batch_size=128,
    epochs=300,
    lr=1e-3,
    hint_rate=0.9,
    hint2_rate=0.6,
    alpha=100,
    l1_lambda=1.0,
    beta=5.0,
    curriculum=50,
    eps_zero=0.2,
    gp_weight=10.0,
    recon_loss="mse",
    device=None
):
    """
    Trains SmartImpute:
      X: [N_cells×G_genes] tensor (normalized counts)
      M: same shape, 1=observed, 0=true zero
    Returns:
      G_model, history, imputed_matrix (numpy G×N)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device); M = M.to(device)

    N, G = X.shape
    G_model = Generator(G).to(device)
    D1 = D1_WGAN_GP(G).to(device)
    D2 = D2_BCE(G).to(device)

    optG  = torch.optim.Adam(G_model.parameters(), lr=lr)
    optD1 = torch.optim.Adam(D1.parameters(),        lr=lr)
    optD2 = torch.optim.Adam(D2.parameters(),        lr=lr)

    ds = TensorDataset(X, M)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    for epoch in range(1, epochs+1):
        sumD1 = sumD2 = sumG = sumRec = sumL1 = 0.0

        for x_mb, m_mb in dl:
            # 1) D1 ⮕ WGAN-GP
            hint1 = m_mb * torch.bernoulli(torch.full_like(m_mb, hint_rate))
            # inject noise & impute
            z_mb  = torch.rand_like(x_mb)
            x_hat = m_mb * x_mb + (1-m_mb) * z_mb
            g_pred = G_model(x_hat, m_mb)
            hat_X  = m_mb * x_mb + (1-m_mb) * g_pred

            # D1 real/fake scores
            d1_real = D1(x_mb, hint1)
            d1_fake = D1(hat_X.detach(), hint1)
            loss_d1 = d1_fake.mean() - d1_real.mean()
            gp      = gradient_penalty(D1, x_mb, hat_X.detach(), hint1, gp_weight)
            optD1.zero_grad()
            (loss_d1 + gp).backward()
            optD1.step()
            sumD1 += (loss_d1 + gp).item()

            # 2) D2 ⮕ BCE on thresholded zeros
            loss_d2 = 0.0
            if epoch > curriculum:
                zero_mask = (hat_X.detach().abs() < eps_zero)
                if zero_mask.any():
                    hint2 = zero_mask.float() * torch.bernoulli(torch.full_like(zero_mask, hint2_rate))
                    prob  = D2(hat_X.detach(), hint2)
                    # create a forced-mask by sampling half of non-zero true outputs
                    forced = torch.zeros_like(zero_mask, dtype=torch.float)
                    nonzero_idx = (~zero_mask).nonzero(as_tuple=False)
                    # sample 50% to force-drop
                    sel = nonzero_idx[torch.randperm(nonzero_idx.size(0))[:nonzero_idx.size(0)//2]]
                    forced[sel[:,0], sel[:,1]] = 1.0
                    pmask = zero_mask & (forced==1)
                    if pmask.any():
                        loss_d2 = F.binary_cross_entropy(prob[pmask], forced[pmask])
                        optD2.zero_grad(); loss_d2.backward(); optD2.step()
                        sumD2 += loss_d2.item()

            # 3) G update
            # adversarial
            g_adv = -D1(hat_X, hint1).mean()
            # reconstruction + L1
            rec = recon_loss_fn(x_mb, g_pred, m_mb, recon_loss)
            l1  = torch.mean(torch.abs((1-m_mb) * g_pred))
            # D2 penalty
            g2  = 0.0
            if epoch > curriculum and 'pmask' in locals() and pmask.any():
                prob_gen = D2(hat_X, hint2)
                g2 = F.binary_cross_entropy(prob_gen[pmask], torch.zeros_like(forced[pmask]))

            G_loss = g_adv + alpha*rec + l1_lambda*l1 + beta*g2
            optG.zero_grad(); G_loss.backward(); optG.step()
            sumG   += g_adv.item()
            sumRec += rec.item()
            sumL1  += l1.item()

        hist = {
            "epoch": epoch,
            "D1":   sumD1/len(dl),
            "D2":   sumD2/len(dl) if epoch>curriculum else 0.0,
            "G_adv":sumG/len(dl),
            "Rec":  sumRec/len(dl),
            "L1":   sumL1/len(dl)
        }
        history.append(hist)
        if epoch%10==0 or epoch==1:
            print(f"[{epoch:03d}] D1={hist['D1']:.4f} D2={hist['D2']:.4f} "
                  f"G_adv={hist['G_adv']:.4f} Rec={hist['Rec']:.4f} L1={hist['L1']:.4f}")

    # final imputation
    with torch.no_grad():
        Z = torch.rand_like(X)
        mu = G_model(M*X + (1-M)*Z, M)
        imputed = (M*X + (1-M)*mu).cpu().numpy()

    return G_model, history, imputed
