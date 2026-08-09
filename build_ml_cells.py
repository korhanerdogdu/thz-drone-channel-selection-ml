"""Rebuilds ML sections (N-Q) in the notebook.
Strips any existing cells from index 35 onward, then appends fresh ML sections.
"""
import nbformat

nb_path = r'C:\Users\Msi\Desktop\sabancı_ders\ens_paper_codebase\thz_maxactive_reproduction.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Keep only the MaxActive reference cells (first 35)
nb.cells = nb.cells[:35]

nb.metadata['kernelspec'] = {
    "display_name": "Python 3.13 (system)",
    "language": "python",
    "name": "python313"
}

def md(text):
    return nbformat.v4.new_markdown_cell(text)

def code(text):
    return nbformat.v4.new_code_cell(text)

new_cells = []

# =============================================================================
# SECTION N — DATA PREPARATION
# =============================================================================
new_cells.append(md(
    "---\n## Section N \u2013 Data Preparation for ML / D3QN\n\n"
    "Build a dataset of **(state, MaxActive-WF, MaxActive-EP)** tuples "
    "for all three scenarios and all 11 distance steps.\n\n"
    "State: the 0.75\u20134.4 THz band is grouped into **Nmax = 20 sub-bands**; "
    "each sub-band contributes mean path loss and log-noise \u2192 state \u2208 R^(2\u00d7Nmax)."
))

new_cells.append(code(
'''import numpy as np
import warnings, time
warnings.filterwarnings('ignore')

# Sub-band parameters
Nmax      = 20          # frequency sub-bands
amax      = 10          # D3QN discrete actions
FREQ_LOW  = StartFreqTHz
FREQ_HIGH = StopFreqTHz

def build_subband_state(Freq_THz, PLtotal_dB, AbsLoss_dB):
    idx = np.where((Freq_THz >= FREQ_LOW) & (Freq_THz <= FREQ_HIGH))[0]
    F   = Freq_THz[idx]
    PL  = PLtotal_dB[idx]
    Ab  = AbsLoss_dB[idx]
    dF  = delta_freq_hz(Freq_THz)[idx]

    edges        = np.linspace(FREQ_LOW, FREQ_HIGH, Nmax + 1)
    sb_mean_PL   = np.zeros(Nmax)
    sb_log_noise = np.zeros(Nmax)
    sb_bandwidths= np.zeros(Nmax)
    sb_valid     = np.zeros(Nmax, dtype=bool)

    for k in range(Nmax):
        m = (F >= edges[k]) & (F < edges[k+1])
        if m.sum() == 0:
            continue
        sb_mean_PL[k]    = PL[m].mean()
        Pn_k             = noise_power(Ab[m], dF[m])
        sb_log_noise[k]  = np.log10(np.maximum(Pn_k.mean(), 1e-40))
        sb_bandwidths[k] = dF[m].sum()
        sb_valid[k]      = True

    state = np.concatenate([sb_mean_PL, sb_log_noise]).astype(np.float32)
    return state, edges, sb_mean_PL, sb_log_noise, sb_bandwidths, sb_valid


def eval_subband_selection(selected_sb_mask, Freq_THz, PLtotal_dB,
                            AbsLoss_dB, Gain_dB, power_alloc='EP'):
    edges   = np.linspace(FREQ_LOW, FREQ_HIGH, Nmax + 1)
    idx_all = np.where((Freq_THz >= FREQ_LOW) & (Freq_THz <= FREQ_HIGH))[0]
    F       = Freq_THz[idx_all]

    ch_mask = np.zeros(len(Freq_THz), dtype=int)
    for k in np.where(selected_sb_mask)[0]:
        in_sb = (F >= edges[k]) & (F < edges[k+1])
        ch_mask[idx_all[in_sb]] = 1

    if ch_mask.sum() == 0:
        return 0.0

    PT  = db2pow(TxPower_dBW + Gain_dB)
    PNB = np.where(ch_mask == 1, pow2db(PT), -np.inf)
    cap, _, _ = mask_cap(ch_mask, Freq_THz, PLtotal_dB, AbsLoss_dB, 0.0, PNB)
    return cap


print(f'Sub-band setup: Nmax={Nmax}, amax={amax}, freq {FREQ_LOW}-{FREQ_HIGH} THz')
Ft, Ab, PL = load_lblrtm(0.001)
s, edges, sb_PL, sb_noise, sb_bw, sb_v = build_subband_state(Ft, PL, Ab)
print(f'State vector shape: {s.shape}   |   Valid sub-bands: {sb_v.sum()}/{Nmax}')
'''
))

new_cells.append(code(
'''# Generate reference dataset
print('Generating MaxActive reference dataset...')

dataset = []

scenarios = [
    ('aligned',    x_r_aligned, y_r_aligned, z_r_aligned, 0, 0),
    ('diag_noAdj', x_r_diag,    y_r_diag,    z_r_diag,    0, 1),
    ('diag_Adj',   x_r_diag,    y_r_diag,    z_r_diag,    1, 2),
]

ref_caps = {
    ('aligned',    'WF'): cap_aligned_WF,
    ('aligned',    'EP'): cap_aligned_EP,
    ('diag_noAdj', 'WF'): cap_diag_noAdj_WF,
    ('diag_noAdj', 'EP'): cap_diag_noAdj_EP,
    ('diag_Adj',   'WF'): cap_diag_Adj_WF,
    ('diag_Adj',   'EP'): cap_diag_Adj_EP,
}

for (sc_name, xr, yr, zr, bwo, sc_id) in scenarios:
    for k in range(N_STEPS):
        CoordRx_k = np.array([xr[k], yr[k], zr[k]])
        d3        = float(np.sqrt(np.sum((CoordRx_k - CoordTx_init)**2)))
        range_km  = d3 / 1000.0

        Freq_THz, AbsLoss_dB, PLtotal_dB = load_lblrtm(range_km)

        if bwo == 0:
            bw_u = BeamWidth_deg
        else:
            bw_u, _ = antenna_gain_optim(
                BeamWidth_deg, SideLobeStrength, TxPower_dBmW,
                CoordRx_k, CoordTx_init,
                tTxBS, pTxBS, tRxBS, pRxBS,
                ((Freq_THz >= StartFreqTHz) & (Freq_THz <= StopFreqTHz)).astype(int),
                PLtotal_dB, AbsLoss_dB, Freq_THz
            )
        Gain_dB = total_antenna_gain_dB(
            CoordTx_init, CoordRx_k,
            tTxBS, pTxBS, tRxBS, pRxBS,
            bw_u, bw_u, SideLobeStrength, SideLobeStrength
        )

        state, edges, sb_PL, sb_noise, sb_bw, sb_valid = build_subband_state(
            Freq_THz, PLtotal_dB, AbsLoss_dB
        )

        dataset.append({
            'scenario':    sc_name,
            'scenario_id': sc_id,
            'step':        k,
            'dist_nom':    dist_axis[k],
            'dist_3d':     d3,
            'Gain_dB':     Gain_dB,
            'bw_used':     bw_u,
            'state':       state,
            'sb_PL':       sb_PL,
            'sb_noise':    sb_noise,
            'sb_bw':       sb_bw,
            'sb_valid':    sb_valid,
            'Freq_THz':    Freq_THz,
            'PLtotal_dB':  PLtotal_dB,
            'AbsLoss_dB':  AbsLoss_dB,
            'cap_MA_WF':   ref_caps[(sc_name, 'WF')][k],
            'cap_MA_EP':   ref_caps[(sc_name, 'EP')][k],
        })

print(f'Dataset: {len(dataset)} samples  '
      f'({len(dataset)//N_STEPS} scenarios x {N_STEPS} steps)')
print(f'State dimension: {dataset[0]["state"].shape[0]}  (2 x {Nmax} sub-bands)')
'''
))

# =============================================================================
# SECTION O — D3QN
# =============================================================================
new_cells.append(md(
    "## Section O \u2013 D3QN Implementation\n\n"
    "Implements the D3QN channel-selection algorithm from CLAUDE.md:\n"
    "- **Dueling DQN** (V-stream + A-stream)\n"
    "- **Double DQN** (online + target network)\n"
    "- **Experience Replay** buffer\n"
    "- **Reward:** `ln((Cmax-Cmin)+1) - ln((Cmax-C)+1)`\n"
    "- D3QN uses equal-power (EP) allocation \u2192 compared against MaxActive EP"
))

new_cells.append(code(
'''import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=(128, 64)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(hidden[1], 32), nn.ReLU(), nn.Linear(32, 1))
        self.adv_fc   = nn.Sequential(
            nn.Linear(hidden[1], 32), nn.ReLU(), nn.Linear(32, n_actions))

    def forward(self, x):
        h = self.shared(x)
        V = self.value_fc(h)
        A = self.adv_fc(h)
        return V + (A - A.mean(dim=-1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


def d3qn_select_channels(state_vec, model, sample_data):
    s_t    = torch.FloatTensor(state_vec).unsqueeze(0)
    with torch.no_grad():
        Q = model(s_t).squeeze(0).numpy()
    action = int(np.argmax(Q))
    N_sel  = max(1, int(np.ceil((action + 1) / amax * Nmax)))

    PL_lin    = db2pow(sample_data['sb_PL'])
    Noise_lin = 10.0 ** sample_data['sb_noise']
    dummy_SNR = np.where(sample_data['sb_valid'],
                         1.0 / (PL_lin * Noise_lin + 1e-30), 0.0)

    sorted_sb = np.argsort(dummy_SNR)[::-1]
    selected  = np.zeros(Nmax, dtype=bool)
    selected[sorted_sb[:N_sel]] = True

    cap = eval_subband_selection(
        selected,
        sample_data['Freq_THz'], sample_data['PLtotal_dB'],
        sample_data['AbsLoss_dB'], sample_data['Gain_dB'], power_alloc='EP'
    )
    return cap, N_sel, selected, action


state_dim  = 2 * Nmax
n_actions  = amax
online_net = DuelingDQN(state_dim, n_actions).to(device)
target_net = DuelingDQN(state_dim, n_actions).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

s_test = torch.FloatTensor(dataset[0]['state']).unsqueeze(0)
print(f'DuelingDQN OK | Q-shape: {online_net(s_test).shape} | '
      f'State dim: {state_dim} | Actions: {n_actions}')
'''
))

new_cells.append(code(
'''# D3QN Training Loop
diag_states     = [s for s in dataset if s['scenario'] == 'diag_noAdj']
diag_adj_states = [s for s in dataset if s['scenario'] == 'diag_Adj']
train_states    = diag_states + diag_adj_states

LR            = 5e-4
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.992
BATCH_SIZE    = 32
TARGET_UPDATE = 15
N_EPISODES    = 600

online_net = DuelingDQN(state_dim, n_actions, hidden=(128, 64)).to(device)
target_net = DuelingDQN(state_dim, n_actions, hidden=(128, 64)).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()
optimizer  = optim.Adam(online_net.parameters(), lr=LR)
criterion  = nn.SmoothL1Loss()
replay_buf = ReplayBuffer(capacity=2000)
epsilon    = EPSILON_START

ep_rewards    = []; ep_capacities = []; ep_ma_reference = []; ep_losses = []

def compute_reward(C, Cmax_ref, Cmin=1e3):
    Cmax = max(Cmax_ref, C, Cmin + 1)
    return float(np.log(max((Cmax - Cmin)+1, 1e-10)) -
                 np.log(max((Cmax - C)   +1, 1e-10)))

print(f'Training D3QN for {N_EPISODES} episodes on {len(train_states)} states ...')
t0 = time.time()

for ep in range(N_EPISODES):
    ep_r = []; ep_c = []; ep_ma = []
    random.shuffle(train_states)

    for sample in train_states:
        s        = sample['state']
        Cmax_ref = max(sample['cap_MA_WF'], sample['cap_MA_EP'])
        Cmin_ref = min(sample['cap_MA_EP'] * 0.1, 1e3)

        if random.random() < epsilon:
            a = random.randint(0, amax - 1)
        else:
            with torch.no_grad():
                a = int(online_net(torch.FloatTensor(s).unsqueeze(0)).argmax().item())

        N_sel = max(1, int(np.ceil((a+1) / amax * Nmax)))
        PL_l  = db2pow(sample['sb_PL'])
        N_l   = 10.0 ** sample['sb_noise']
        dSNR  = np.where(sample['sb_valid'], 1.0/(PL_l*N_l+1e-30), 0.0)
        sel   = np.zeros(Nmax, dtype=bool)
        sel[np.argsort(dSNR)[::-1][:N_sel]] = True

        C  = eval_subband_selection(sel, sample['Freq_THz'], sample['PLtotal_dB'],
                                    sample['AbsLoss_dB'], sample['Gain_dB'])
        r  = compute_reward(C, Cmax_ref, Cmin_ref)
        s2 = random.choice(train_states)['state']
        replay_buf.push(s, a, r, s2, 0.0)
        ep_r.append(r); ep_c.append(C); ep_ma.append(Cmax_ref)

    loss_val = 0.0
    if len(replay_buf) >= BATCH_SIZE:
        sb, ab, rb, s2b, db = replay_buf.sample(BATCH_SIZE)
        with torch.no_grad():
            next_a  = online_net(s2b).argmax(1, keepdim=True)
            next_Q  = target_net(s2b).gather(1, next_a).squeeze(1)
            targets = rb + GAMMA * next_Q * (1 - db)
        curr_Q = online_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
        loss   = criterion(curr_Q, targets)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
        optimizer.step()
        loss_val = float(loss.item())

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    ep_rewards.append(np.mean(ep_r))
    ep_capacities.append(np.mean(ep_c))
    ep_ma_reference.append(np.mean(ep_ma))
    ep_losses.append(loss_val)

    if (ep+1) % 100 == 0:
        ma   = np.mean(ep_ma)/1e9
        d3qn = np.mean(ep_c)/1e9
        print(f'  Ep {ep+1:4d} | eps={epsilon:.3f} | '
              f'Rew={np.mean(ep_r):.3f} | D3QN={d3qn:.4f} Gbps | '
              f'MA={ma:.4f} Gbps | Ratio={d3qn/(ma+1e-12)*100:.1f}% | '
              f'Loss={loss_val:.4f}')

print(f'Done in {time.time()-t0:.1f}s  |  final eps={epsilon:.4f}')
'''
))

new_cells.append(code(
'''# D3QN Convergence Plot
def smooth(arr, w=20):
    out = np.convolve(arr, np.ones(w)/w, mode='valid')
    return np.arange(len(out)), out

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('D3QN Training Convergence', fontsize=13)

ax = axes[0,0]
x, y = smooth(ep_rewards, 20)
ax.plot(ep_rewards, alpha=0.3, color='steelblue')
ax.plot(x, y, color='steelblue', lw=2)
ax.set_title('Mean Reward per Episode'); ax.set_xlabel('Episode')
ax.set_ylabel('Reward'); ax.grid(True, alpha=0.4)

ax = axes[0,1]
cap_gbps = np.array(ep_capacities)/1e9
ref_gbps = np.array(ep_ma_reference)/1e9
x, yc = smooth(cap_gbps, 20)
ax.plot(cap_gbps, alpha=0.3, color='orange')
ax.plot(x, yc, color='orange', lw=2, label='D3QN capacity')
ax.plot(ref_gbps, color='green', lw=1.5, ls='--', label='MaxActive ref')
ax.set_title('Capacity: D3QN vs MaxActive'); ax.set_xlabel('Episode')
ax.set_ylabel('Capacity [Gbps]'); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

ax = axes[1,0]
ratio = np.array(ep_capacities)/(np.array(ep_ma_reference)+1e-12)*100
x, yr = smooth(ratio, 20)
ax.plot(ratio, alpha=0.3, color='purple')
ax.plot(x, yr, color='purple', lw=2)
ax.axhline(100, ls='--', color='black', lw=1)
ax.set_title('D3QN / MaxActive Ratio (%)'); ax.set_xlabel('Episode')
ax.set_ylabel('Ratio (%)'); ax.grid(True, alpha=0.4)

ax = axes[1,1]
x, yl = smooth(ep_losses, 20)
ax.plot(ep_losses, alpha=0.3, color='red')
ax.plot(x, yl, color='red', lw=2)
ax.set_title('Training Loss (Huber)'); ax.set_xlabel('Episode')
ax.set_ylabel('Loss'); ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('d3qn_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: d3qn_convergence.png')
'''
))

new_cells.append(code(
'''# Evaluate D3QN on all three scenarios
online_net.eval()
d3qn_results = {}   # sc_name -> EP-style capacity array

sc_order  = ['aligned', 'diag_noAdj', 'diag_Adj']
sc_titles = ['(a) Aligned', '(b) Misaligned, no BW adj', '(c) Misaligned, with BW adj']

for sc_name in sc_order:
    sc_samp = sorted([s for s in dataset if s['scenario'] == sc_name],
                     key=lambda s: s['step'])
    caps = []
    for sample in sc_samp:
        cap, _, _, _ = d3qn_select_channels(sample['state'], online_net, sample)
        caps.append(cap)
    d3qn_results[sc_name] = np.array(caps)
    ma_ep = np.array([s['cap_MA_EP'] for s in sc_samp])
    ratio = d3qn_results[sc_name] / (ma_ep + 1e-12) * 100
    print(f'D3QN {sc_name:12s}: mean {np.mean(d3qn_results[sc_name])/1e9:.4f} Gbps '
          f'| ratio vs MA-EP: {ratio.mean():.1f}%')
'''
))

# =============================================================================
# SECTION P — SUPERVISED ML BASELINES
# =============================================================================
new_cells.append(md(
    "## Section P \u2013 Supervised ML Baselines\n\n"
    "Models trained **separately** for MaxActive WF and MaxActive EP targets:\n\n"
    "**Ridge, ElasticNet, Decision Tree, K-NN, MLP, Random Forest, XGBoost, "
    "Gradient Boosting**\n\n"
    "**Features (10):** log-distance, gain (dB), norm. beamwidth, scenario id, "
    "alloc flag, mean PL, std PL, mean log-noise, std log-noise, valid fraction.\n\n"
    "**Target:** log10(capacity in Gbps). **Split:** 75% train / 25% test, stratified."
))

new_cells.append(code(
'''from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import Ridge, ElasticNet
from sklearn.tree           import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble       import (RandomForestRegressor,
                                    GradientBoostingRegressor)
from sklearn.neighbors      import KNeighborsRegressor
from sklearn.model_selection import (GridSearchCV, StratifiedShuffleSplit, KFold)
from sklearn.metrics        import (mean_squared_error, mean_absolute_error,
                                    r2_score)
import xgboost as xgb

def build_features(sample, alloc='WF'):
    vl = sample['sb_valid']
    pl = sample['sb_PL'][vl]
    nn = sample['sb_noise'][vl]
    return np.array([
        np.log10(sample['dist_3d'] + 1),
        sample['Gain_dB'],
        sample['bw_used'] / 360.0,
        float(sample['scenario_id']),
        1.0 if alloc == 'WF' else 0.0,
        pl.mean(), pl.std(),
        nn.mean(), nn.std(),
        float(vl.sum()) / Nmax,
    ], dtype=np.float32)

X_rows, y_WF, y_EP, scen_labels = [], [], [], []
for s in dataset:
    X_rows.append(build_features(s, 'WF'))
    y_WF.append(s['cap_MA_WF'])
    y_EP.append(s['cap_MA_EP'])
    scen_labels.append(s['scenario_id'])

X      = np.array(X_rows)
yW     = np.log10(np.maximum(np.array(y_WF)/1e9, 1e-12))
yE     = np.log10(np.maximum(np.array(y_EP)/1e9, 1e-12))
labels = np.array(scen_labels)

print(f'Feature matrix: {X.shape}')
print(f'WF target log10(Gbps): [{yW.min():.2f}, {yW.max():.2f}]')
print(f'EP target log10(Gbps): [{yE.min():.2f}, {yE.max():.2f}]')
'''
))

new_cells.append(code(
'''# Train / Test split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, labels))

X_train, X_test = X[train_idx], X[test_idx]
yW_tr, yW_te    = yW[train_idx], yW[test_idx]
yE_tr, yE_te    = yE[train_idx], yE[test_idx]
lbl_te          = labels[test_idx]

print(f'Train: {len(train_idx)} | Test: {len(test_idx)}')
'''
))

new_cells.append(code(
'''# Training helpers
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_preds(y_te, pred, name=''):
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    r2   = float(r2_score(y_te, pred))
    mape = float(np.mean(np.abs(10**pred - 10**y_te)/(10**y_te+1e-12))*100)
    if name:
        print(f'  {name:30s}  RMSE={rmse:.3f}  MAE={mae:.3f}  '
              f'R2={r2:.3f}  MAPE={mape:.1f}%')
    return dict(rmse=rmse, mae=mae, r2=r2, mape=mape)


def fit_model(name, estimator, X_tr, y_tr, X_te, y_te, scale=True):
    sc   = StandardScaler() if scale else None
    Xtr  = sc.fit_transform(X_tr) if sc else X_tr
    Xte  = sc.transform(X_te)     if sc else X_te
    estimator.fit(Xtr, y_tr)
    pred     = estimator.predict(Xte)
    pred_all = estimator.predict(sc.transform(X) if sc else X)
    m = eval_preds(y_te, pred, name)
    return dict(name=name, model=estimator, scaler=sc,
                pred=pred, pred_all=pred_all, **m)


def gs_fit(name, base, param_grid, X_tr, y_tr, X_te, y_te, scale=True):
    sc   = StandardScaler() if scale else None
    Xtr  = sc.fit_transform(X_tr) if sc else X_tr
    Xte  = sc.transform(X_te)     if sc else X_te
    gs   = GridSearchCV(base, param_grid, cv=cv,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    gs.fit(Xtr, y_tr)
    best     = gs.best_estimator_
    pred     = best.predict(Xte)
    pred_all = best.predict(sc.transform(X) if sc else X)
    m = eval_preds(y_te, pred, name)
    print(f'    best params: {gs.best_params_}')
    return dict(name=name, model=best, scaler=sc,
                pred=pred, pred_all=pred_all, **m)


# Train WF models
print('=== MaxActive WF target ===')
results_WF = {}

results_WF['Ridge'] = fit_model(
    'Ridge', Ridge(alpha=1.0), X_train, yW_tr, X_test, yW_te)

results_WF['ElasticNet'] = fit_model(
    'ElasticNet', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    X_train, yW_tr, X_test, yW_te)

results_WF['DecisionTree'] = gs_fit(
    'DecisionTree', DecisionTreeRegressor(random_state=42),
    {'max_depth': [2,3,5,None], 'min_samples_split': [2,4,8]},
    X_train, yW_tr, X_test, yW_te, scale=False)

results_WF['KNN'] = gs_fit(
    'KNN', KNeighborsRegressor(),
    {'n_neighbors': [2,3,5], 'weights': ['uniform','distance']},
    X_train, yW_tr, X_test, yW_te)

results_WF['MLP'] = gs_fit(
    'MLP',
    MLPRegressor(max_iter=2000, early_stopping=True,
                 validation_fraction=0.15, random_state=42),
    {'hidden_layer_sizes': [(64,32),(128,64),(128,64,32)],
     'alpha': [1e-4,1e-3,1e-2], 'learning_rate_init': [1e-3,5e-4]},
    X_train, yW_tr, X_test, yW_te)

results_WF['RandomForest'] = gs_fit(
    'RandomForest', RandomForestRegressor(random_state=42),
    {'n_estimators': [100,200,500], 'max_depth': [None,5,10],
     'min_samples_split': [2,5]},
    X_train, yW_tr, X_test, yW_te, scale=False)

results_WF['XGBoost'] = gs_fit(
    'XGBoost', xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
    {'n_estimators': [100,200,500], 'max_depth': [3,5,7],
     'learning_rate': [0.01,0.05,0.1], 'subsample': [0.8,1.0]},
    X_train, yW_tr, X_test, yW_te, scale=False)

results_WF['GradBoost'] = gs_fit(
    'GradBoost', GradientBoostingRegressor(random_state=42),
    {'n_estimators': [100,200], 'max_depth': [3,5],
     'learning_rate': [0.05,0.1,0.2]},
    X_train, yW_tr, X_test, yW_te, scale=False)
'''
))

new_cells.append(code(
'''# Train EP models
print('=== MaxActive EP target ===')
results_EP = {}

results_EP['Ridge'] = fit_model(
    'Ridge', Ridge(alpha=1.0), X_train, yE_tr, X_test, yE_te)

results_EP['ElasticNet'] = fit_model(
    'ElasticNet', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    X_train, yE_tr, X_test, yE_te)

results_EP['DecisionTree'] = gs_fit(
    'DecisionTree', DecisionTreeRegressor(random_state=42),
    {'max_depth': [2,3,5,None], 'min_samples_split': [2,4,8]},
    X_train, yE_tr, X_test, yE_te, scale=False)

results_EP['KNN'] = gs_fit(
    'KNN', KNeighborsRegressor(),
    {'n_neighbors': [2,3,5], 'weights': ['uniform','distance']},
    X_train, yE_tr, X_test, yE_te)

results_EP['MLP'] = gs_fit(
    'MLP',
    MLPRegressor(max_iter=2000, early_stopping=True,
                 validation_fraction=0.15, random_state=42),
    {'hidden_layer_sizes': [(64,32),(128,64),(128,64,32)],
     'alpha': [1e-4,1e-3,1e-2], 'learning_rate_init': [1e-3,5e-4]},
    X_train, yE_tr, X_test, yE_te)

results_EP['RandomForest'] = gs_fit(
    'RandomForest', RandomForestRegressor(random_state=42),
    {'n_estimators': [100,200,500], 'max_depth': [None,5,10],
     'min_samples_split': [2,5]},
    X_train, yE_tr, X_test, yE_te, scale=False)

results_EP['XGBoost'] = gs_fit(
    'XGBoost', xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
    {'n_estimators': [100,200,500], 'max_depth': [3,5,7],
     'learning_rate': [0.01,0.05,0.1], 'subsample': [0.8,1.0]},
    X_train, yE_tr, X_test, yE_te, scale=False)

results_EP['GradBoost'] = gs_fit(
    'GradBoost', GradientBoostingRegressor(random_state=42),
    {'n_estimators': [100,200], 'max_depth': [3,5],
     'learning_rate': [0.05,0.1,0.2]},
    X_train, yE_tr, X_test, yE_te, scale=False)
'''
))

# =============================================================================
# SECTION Q — EVALUATION AND PLOTS
# =============================================================================
new_cells.append(md(
    "## Section Q \u2013 Evaluation, Comparison Plots, and Summary\n\n"
    "**Main figure:** 3 scenarios \u00d7 2 allocations = 6 subplots.  \n"
    "- Left column: MaxActive **WF** true curve + ML-WF predictions  \n"
    "- Right column: MaxActive **EP** true curve + ML-EP predictions + D3QN  \n\n"
    "Additional: metrics bar charts, scatter plots, feature importances, summary table."
))

new_cells.append(code(
'''# Colour/marker style per model
STYLE = {
    'Ridge':        ('#4878D0', 'P--',  1.4),
    'ElasticNet':   ('#EE854A', 'X--',  1.4),
    'DecisionTree': ('#6ACC65', 'D:',   1.4),
    'KNN':          ('#D65F5F', 'v:',   1.4),
    'MLP':          ('#956CB4', 'h-.',  1.4),
    'RandomForest': ('#8C613C', 's-.',  1.4),
    'XGBoost':      ('#DC7EC0', '^--',  1.4),
    'GradBoost':    ('#797979', 'o:',   1.4),
    'D3QN':         ('#17becf', '*-',   2.4),
}

def predict_curve(res, sc_name, alloc='WF'):
    sc_samp = sorted([s for s in dataset if s['scenario'] == sc_name],
                     key=lambda s: s['step'])
    X_sc = np.array([build_features(s, alloc) for s in sc_samp])
    sc   = res['scaler']
    Xsc  = sc.transform(X_sc) if sc else X_sc
    return 10 ** res['model'].predict(Xsc)
'''
))

new_cells.append(code(
'''# Main figure: 3 scenarios x 2 allocations (WF left, EP right)
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
fig.suptitle('All Methods vs MaxActive  |  WF (left column) and EP (right column)',
             fontsize=13, y=1.005)

for row, (sc_name, sc_title) in enumerate(zip(sc_order, sc_titles)):
    sc_samp  = sorted([s for s in dataset if s['scenario'] == sc_name],
                      key=lambda s: s['step'])
    y_ma_wf  = np.array([s['cap_MA_WF']/1e9 for s in sc_samp])
    y_ma_ep  = np.array([s['cap_MA_EP']/1e9 for s in sc_samp])

    for col, (alloc, y_ma, results) in enumerate([
            ('WF', y_ma_wf, results_WF),
            ('EP', y_ma_ep, results_EP)]):

        ax = axes[row, col]

        # MaxActive ground truth
        ax.semilogy(dist_axis, np.maximum(y_ma, 1e-12),
                    'k-o', lw=2.5, ms=8, zorder=10,
                    label=f'MaxActive {alloc} (true)')

        # ML model predictions
        for mname, res in results.items():
            y_pred = predict_curve(res, sc_name, alloc)
            c, mk, lw = STYLE.get(mname, ('gray','o-',1.2))
            ax.semilogy(dist_axis, np.maximum(y_pred, 1e-12),
                        mk, color=c, lw=lw, ms=5, alpha=0.85, label=mname)

        # D3QN on EP column (uses equal-power allocation)
        if col == 1:
            c, mk, lw = STYLE['D3QN']
            ax.semilogy(dist_axis,
                        np.maximum(d3qn_results[sc_name]/1e9, 1e-12),
                        mk, color=c, lw=lw, ms=9, alpha=0.95, label='D3QN')

        ax.set_title(f'{sc_title} — MaxActive {alloc}', fontsize=10)
        ax.set_xlabel('Distance (d) [m]')
        ax.set_ylabel('Capacity (C) [Gbps]')
        ax.set_xticks(dist_axis)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=7.5, loc='upper right', ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.35)

plt.tight_layout()
plt.savefig('all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: all_methods_comparison.png')
'''
))

new_cells.append(code(
'''# Metrics bar charts (WF top row, EP bottom row)
model_names = list(results_WF.keys())
colors      = [STYLE[m][0] for m in model_names]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('ML Model Metrics — WF (top) vs EP (bottom)', fontsize=13)

for row, (results, label) in enumerate([(results_WF,'WF'), (results_EP,'EP')]):
    rmse_v = [results[m]['rmse'] for m in model_names]
    mape_v = [results[m]['mape'] for m in model_names]
    r2_v   = [results[m]['r2']   for m in model_names]

    ax = axes[row, 0]
    ax.bar(model_names, rmse_v, color=colors, alpha=0.85)
    ax.set_title(f'{label}: RMSE (log10 Gbps)'); ax.set_ylabel('RMSE')
    ax.tick_params(axis='x', rotation=35); ax.grid(True, axis='y', alpha=0.4)

    ax = axes[row, 1]
    ax.bar(model_names, mape_v, color=colors, alpha=0.85)
    ax.set_title(f'{label}: MAPE (%)'); ax.set_ylabel('MAPE (%)')
    ax.tick_params(axis='x', rotation=35); ax.grid(True, axis='y', alpha=0.4)

    ax = axes[row, 2]
    ax.bar(model_names, r2_v, color=colors, alpha=0.85)
    ax.axhline(1.0, ls='--', color='black', lw=1)
    ax.set_title(f'{label}: R\u00b2 Score'); ax.set_ylabel('R\u00b2')
    ax.tick_params(axis='x', rotation=35); ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('ml_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: ml_metrics_comparison.png')
'''
))

new_cells.append(code(
'''# Scatter plots: predicted vs actual (WF)
n = len(results_WF)
ncols = 4; nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
axes = axes.flatten()
fig.suptitle('Predicted vs Actual — MaxActive WF (test set)', fontsize=13)

for i, (mname, res) in enumerate(results_WF.items()):
    ax   = axes[i]
    yt   = yW_te; yp = res['pred']
    mn   = min(yt.min(), yp.min()) - 0.3
    mx   = max(yt.max(), yp.max()) + 0.3
    ax.plot([mn,mx],[mn,mx], 'k--', lw=1.5)
    sc_plot = ax.scatter(yt, yp, c=lbl_te, cmap='viridis', s=60, alpha=0.85, zorder=5)
    plt.colorbar(sc_plot, ax=ax, fraction=0.04, pad=0.04).set_label('Scenario', fontsize=8)
    ax.set_title(f'{mname}  R\u00b2={res["r2"]:.3f}', fontsize=10)
    ax.set_xlabel('Actual log10(Gbps)'); ax.set_ylabel('Predicted')
    ax.grid(True, alpha=0.3)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('ml_scatter_WF.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: ml_scatter_WF.png')
'''
))

new_cells.append(code(
'''# Feature importances (tree-based models)
feat_names  = ['log10_dist','G_dB','bw_norm','scenario','alloc_flag',
               'mean_PL','std_PL','mean_noise','std_noise','valid_frac']
tree_models = [m for m in model_names
               if hasattr(results_WF[m]['model'], 'feature_importances_')]

fig, axes = plt.subplots(1, len(tree_models), figsize=(5*len(tree_models)+1, 5))
if len(tree_models) == 1:
    axes = [axes]
fig.suptitle('Feature Importances (tree-based models, WF target)', fontsize=12)

for ax, mname in zip(axes, tree_models):
    imp = results_WF[mname]['model'].feature_importances_
    c   = STYLE.get(mname, ('steelblue',))[0]
    ax.barh(feat_names, imp, color=c, alpha=0.85)
    ax.set_title(mname); ax.set_xlabel('Importance')
    ax.grid(True, axis='x', alpha=0.4)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: feature_importance.png')
'''
))

new_cells.append(code(
'''# Final summary table
print('=' * 90)
print('FINAL RESULTS SUMMARY')
print('=' * 90)

hdr = f'{"Model":22s} | {"RMSE":>6} | {"MAE":>6} | {"R2":>6} | {"MAPE%":>7}'
sep = '-' * 56

for label, results in [('WF', results_WF), ('EP', results_EP)]:
    print(f'\\n--- MaxActive {label} capacity ---')
    print(hdr); print(sep)
    for mname, res in results.items():
        print(f'{res["name"]:22s} | {res["rmse"]:6.3f} | {res["mae"]:6.3f} | '
              f'{res["r2"]:6.3f} | {res["mape"]:7.1f}')

print('\\n--- D3QN vs MaxActive EP ---')
for sc_name, sc_title in zip(sc_order, sc_titles):
    sc_samp = sorted([s for s in dataset if s['scenario'] == sc_name],
                     key=lambda s: s['step'])
    ma_ep = np.array([s['cap_MA_EP'] for s in sc_samp])
    d3    = d3qn_results[sc_name]
    ratio = d3 / (ma_ep + 1e-12) * 100
    print(f'  {sc_title:30s}: D3QN mean={np.mean(d3)/1e9:.4f} Gbps  '
          f'MA-EP mean={np.mean(ma_ep)/1e9:.4f} Gbps  ratio={ratio.mean():.1f}%')

print('\\n--- Saved figures ---')
for fn in ['maxactive_capacity_3panel.png', 'd3qn_convergence.png',
           'all_methods_comparison.png',    'ml_metrics_comparison.png',
           'ml_scatter_WF.png',             'feature_importance.png']:
    print(f'  {fn}')
print('=' * 90)
'''
))

# =============================================================================
# Write notebook
# =============================================================================
nb.cells.extend(new_cells)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Added {len(new_cells)} new cells.')
print(f'Total cells: {len(nb.cells)}')
