import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import matplotlib.pyplot as plt  # type: ignore


# GPU check
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")

if device.type == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

else: 
    print("no gpu") 


# SSM layer
class SSMLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        #A controls how fast the model forgets, stored as a log so it stays positive (learnable parameter) 
        A = torch.arange(1, d_state+1).float().unsqueeze(0).expand(d_model, -1)
        self.A_log  = nn.Parameter(torch.log(A)) 

        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01) # how much new input to absorb
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01) # how much of memory to read out
        self.D = nn.Parameter(torch.ones(d_model))                   # skip connection (passes input directly through)
       
        self.in_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        u  = self.in_proj(x)

        dA = torch.exp(-torch.exp(self.A_log)) # decay factor, always between 0 and 1
        h  = torch.zeros(B, D, self.d_state, device=x.device) # update hidden state
        ys = [] # read from hidden state

        # At every token t, h is memory, multiply old memory by dA (partial forget) + add new input, read out a value using C
        for t in range(L):
            ut = u[:, t, :]
            h  = dA.unsqueeze(0) * h + ut.unsqueeze(-1) * self.B.unsqueeze(0)
            ys.append((h * self.C.unsqueeze(0)).sum(-1))

        return self.out_proj(torch.stack(ys, dim=1) + self.D * u)


class SCTBuffer(nn.Module):
    def __init__(self, d_model, buffer_size=16, write_every=8):
        super().__init__()
        self.buffer_size = buffer_size # max number of snapshots to keep (16)
        self.write_every = write_every # take a snapshot every 8 tokens

        # write_proj, transforms current hidden state before saving to buffer 
        # linear -> tanh -> linear makes it a small 2 layer network (more expressive)
        self.write_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        
        # read_gate: takes current state + retrieved memory, outputs 0-1 per dimension
        # sigmoid squashes to (0,1) so it acts as a soft on/off switch
        # d_model*2 input because we concatenate h and read together
        self.read_gate  = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Sigmoid())
        
        # query, key, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x):
        B, L, D = x.shape
        slot_list, out = [], [] # slot_list = saved snapshots, out = outputs per token

        for t in range(L): # loop over every token position
            h = x[:, t, :] # grab current token's hidden state - shape (B, D)

            if t % self.write_every == 0:              # every 8 steps, take snapshot
                slot_list.append(self.write_proj(h))   # transform h and save it
                if len(slot_list) > self.buffer_size:  # if buffer is full (more than 16 slots taken)
                    slot_list.pop(0)                   # remove oldest snapshot FIFO

            if slot_list: # if we have at least one snapshot
                buf    = torch.stack(slot_list, dim=1)                                          # stack snapshots in shape (B, num_slots, D)
                q      = self.q_proj(h).unsqueeze(1)                                            # query from current token -> (B, 1, D)
                # dot product between query and all keys, scaled by sqrt(D)
                # softmax turns scores into probabilities, then multiply by values to get weighted mix of slot contents
                scores = (q @ self.k_proj(buf).transpose(-2,-1)) / math.sqrt(D)
                read   = (F.softmax(scores, dim=-1) @ self.v_proj(buf)).squeeze(1)
            else:
                read = torch.zeros_like(h)

            # concatenate current state and retrieved memory, learn how much memory to use
            gate = self.read_gate(torch.cat([h, read], dim=-1))
            
            # final output = current state + gated memory
            # gate = 0, no memory use, gate = 1, use memory
            out.append(h + gate * read)

        # stack all per-token outputs back into a sequence shape (B, L, D)
        return torch.stack(out, dim=1)


# benchmark plain ssm 
class PlainSSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, d_state=16, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SSMLayer(d_model, d_state) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = l(h)
        return self.head(self.norm(h))


# implemented SCT-SSM by adding SCTBuffer after the SSM layers 
class SCTSSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, d_state=16, n_layers=2, buffer_size=16, write_every=8):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SSMLayer(d_model, d_state) for _ in range(n_layers)])
        self.sct    = SCTBuffer(d_model, buffer_size, write_every)
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = l(h)
        return self.head(self.norm(self.sct(h)))


# config 
GAPS    = [40, 80, 120, 200]
SEQ_LEN = 256
N_TRAIN = 3000
BATCH   = 64   # increase to 128 or 256 if you have VRAM to spare
VOCAB   = 32
LR      = 3e-4


# position 0 = the entity the model needs to remember (token number etc)
# position 1 to gap = random garbage tokens to drown out the signal
# position gap+1 = a special QUERY token (always token 2) (asks what was observed at position 0)
# rest is padding
# objective is after seeing QUERY, output token 17, it has to have somehow preserved that across all the noise
def make_batch_gap(gap, batch_size=BATCH, seq_len=SEQ_LEN, vocab=VOCAB):
    QUERY  = 2; PAD = 3
    entity = torch.randint(4, vocab, (batch_size,))
    noise  = torch.randint(4, vocab, (batch_size, gap))
    seq    = torch.cat([
        entity.unsqueeze(1), noise,
        torch.full((batch_size, 1), QUERY),
        torch.full((batch_size, seq_len - gap - 2), PAD),
    ], dim=1)
    return seq.to(device), entity.to(device)


# feeds sequence through the model
# only measures loss at position gap+1 (answer slot)
# backprops, clip gradients and steps, every 200 steps print a progress line
def train_gap(model, gap, name, n_steps=N_TRAIN):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # GradScaler enables mixed precision training — uses float16 on GPU where safe
    # this is ~2x faster than float32 on your RTX 4060, disabled automatically on CPU
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))


    accs = []

    for step in range(n_steps):
        seq, target = make_batch_gap(gap)

        # autocast runs the forward pass in float16 where possible (faster on GPU)
        # loss and backward are handled safely by the scaler
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(seq)[:, gap+1, :]  # only look at position gap+1 (the answer slot)
            loss = F.cross_entropy(pred, target)

        opt.zero_grad()
        scaler.scale(loss).backward()       # scaled backward pass (prevents float16 underflow)
        scaler.unscale_(opt)                # unscale before clipping so clip threshold is correct
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        scaler.step(opt)                    # update weights
        scaler.update()                     # update the scaler for next step

        # at the end returns the average accuracy of the last 5 checkpoints rather than the final
        # single step (smooths out noise in measurement so a stable number is placed in the table
        # and plot rather than a potentially lucky spike from a single step)
        if step % 200 == 0:
            acc = (pred.argmax(-1) == target).float().mean().item()
            accs.append(acc)
            print(f" [{name}] step {step:4d} | loss {loss.item():.3f} | acc {acc:.3f}")

    return sum(accs[-5:]) / 5


# RUN
plain_final, sct_final = [], []

for gap in GAPS:
    print(f"\n{'='*50}  GAP={gap}  {'='*50}")
    plain = PlainSSMModel(VOCAB)
    sct   = SCTSSMModel(VOCAB, buffer_size=16, write_every=max(4, gap//16))

    plain_final.append(train_gap(plain, gap, "Plain"))
    sct_final.append(train_gap(sct, gap, "SCT  "))

    # free VRAM between gap runs so you don't run out of memory across the 4 experiments
    if device.type == "cuda":
        torch.cuda.empty_cache()


# PLOT 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(GAPS, plain_final, 'o-', color='tomato',    label='Plain SSM', linewidth=2, markersize=8)
ax.plot(GAPS, sct_final,   's-', color='steelblue', label='SCT-SSM',   linewidth=2, markersize=8)
ax.axhline(1/VOCAB, color='gray', linestyle='--', label=f'Random ({1/VOCAB:.2f})')

for i, gap in enumerate(GAPS):
    ax.annotate(f'{plain_final[i]:.2f}', (gap, plain_final[i]), textcoords="offset points", xytext=(0,-16), ha='center', color='tomato',    fontsize=9)
    ax.annotate(f'{sct_final[i]:.2f}',   (gap, sct_final[i]),   textcoords="offset points", xytext=(0, 8),  ha='center', color='steelblue', fontsize=9)

ax.set_xlabel('Gap (tokens between entity and query)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('SCT-SSM vs Plain SSM (Accuracy vs Recall Distance)', fontsize=12)
ax.set_ylim(0, 0.8); ax.set_xticks(GAPS); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('gap_scaling.png', dpi=150); plt.show()

print(f"\n{'GAP':<8} {'Plain':>8} {'SCT':>8} {'Delta':>8}")
print()
print()

for i, gap in enumerate(GAPS):
    print(f"{gap:<8} {plain_final[i]:>8.3f} {sct_final[i]:>8.3f} {sct_final[i]-plain_final[i]:>+8.3f}")
print(f"\nRandom: {1/VOCAB:.3f}")