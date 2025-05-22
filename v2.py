import torch
import torch.nn as nn 
from torch.nn import functional as F 

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

print("Length", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary: {''.join(chars)}, Vocab Size: {vocab_size}")

# creating the mapping
itos = {i:s for i,s in enumerate(chars)}
stoi = {s:i for i,s in itos.items()}

encode = lambda l: [stoi[c] for c in l]
decode = lambda l: "".join([itos[c] for c in l])


print(encode('hii there'))
print(decode(encode('hii there')))


# use pytorch to access and store it 
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size + 1] # we gotta do block size + 1 cuz we'll use 8 chars as context to genrate the 9th one    

def get_batch(split):

  data = train_data if split=='train' else val_data 
  ix = torch.randint(0,len(data)-block_size, (batch_size,))
  xb = torch.stack([data[i:i+block_size] for i in ix])
  yb = torch.stack([data[i+1:i+block_size+1] for i in ix])


  return xb, yb 



torch.manual_seed(1337)

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.cat([h(x) for h in self.heads],dim=-1)
    x = self.dropout(self.proj(x))
    return x

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        
        self.weights = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        ) 
    
    def forward(self,x):
        return self.weights(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_emb):
        B,T,C = token_emb.shape
        q = self.q(token_emb) # B T HS
        k = self.k(token_emb) # B T HS
        v = self.v(token_emb) # B T HS

        wei = q @ k.transpose(-2,-1) # B T HS @ B HS T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) * C ** 0.5
        wei = F.softmax(wei,dim=-1) # B T T
        wei = self.dropout(wei)
        xbow = wei @ v # B T T @ B T HS
        return xbow # B T HS

class Block(nn.Module):
  
  def __init__(self, n_embd, num_heads):
    super().__init__()
    head_size = n_embd // num_heads
    self.mha_heads = MultiHeadAttention(head_size, num_heads)
    self.ffw = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.mha_heads(self.ln1(x))
    x = x + self.ffw(self.ln2(x))
    return x
    

class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa_head = SelfAttention(n_embd)
    # self.sa_heads = MultiHeadAttention(num_heads, n_embd//num_heads)
    # self.ffw = FeedForward(n_embd)
    # self.blocks = nn.Sequential(
    #   Block(n_embd, num_heads),
    #   Block(n_embd, num_heads),
    #   Block(n_embd, num_heads),
    #   nn.LayerNorm(n_embd)
    # )
    self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, target=None):
    B,T = idx.shape
    token_emb = self.token_embedding_table(idx) #B,T,n_embd
    positional_output = self.position_embedding_table(torch.tensor(torch.arange(0,T),device=device)) #T,n_embd
    x = token_emb + positional_output # B T n_embd
    # x = self.sa_heads(x) # B, T, HS
    # x = self.ffw(x)
    x = self.blocks(x) # B, T, HS
    x = self.ln_f(x)
    logits = self.lm_head(x) # B,T,vocab_size
    
    
    if target is not None:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      target = target.view(B*T,)
      loss = F.cross_entropy(logits, target)
    else:
      loss = None

    return logits, loss

  def generate(self, idx, max_num_tokens):
    for i in range(max_num_tokens):
      idx_needed = idx[:, -block_size:]
      logits, loss = self(idx_needed) # B x T x C 
      # print(logits.shape)
      logits = logits[:, -1, :] # take last layer dim alone
      probs = F.softmax(logits, dim=-1) # column wise for each batch

      idx_new = torch.multinomial(probs, num_samples=1) 
      idx = torch.cat((idx, idx_new), dim=1) # concat along dim 1, so it becomes Bx T+1
    return idx

@torch.no_grad()
def evaluate_model(model):
    '''
    get train loss, val loss
    '''
    model.eval()
    out_dict = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb,yb = get_batch(split)
            logits, loss = model(xb,yb)
            losses[k] = loss.item()
        out_dict[f'{split}_loss'] = losses.mean().item()
    model.train()
    return out_dict

        



# eval_iters = 500
# batch_size = 64
# eval_interval = 500
# max_steps = 10000
# n_embd = 384
# block_size = 256
# # head_size = 16
# num_heads = 6
# dropout = 0.2
# n_layer = 6

eval_iters = 500
batch_size = 32
eval_interval = 500
max_steps = 10000
n_embd = 128
block_size = 32
# head_size = 16
num_heads = 4
dropout = 0.2
n_layer = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Detected Device", device)



m = BigramLanguageModel().to(device)


# print(sum([p.nelement() for p in m.parameters()]))

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
for step in range(max_steps):
  # print(f"Step: {step}")
  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if step % eval_interval == 0:
      out_dict = evaluate_model(m)
      print(f"Step {step}/{max_steps}: Train Loss {out_dict['train_loss']:.4f}, Val Loss {out_dict['val_loss']:.4f}")

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_num_tokens=500)[0].tolist()))