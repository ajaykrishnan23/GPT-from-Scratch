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



block_size = 8
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
  
  def __init__(self, vocab_size, n_embd):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, target=None):
    B,T = idx.shape
    token_emb = self.token_embedding_table(idx) #B,T,n_embd
    positional_output = self.position_embedding_table(torch.tensor(torch.arange(0,T),device=device)) #T,n_embd
    x = token_emb + positional_output
    logits = self.lm_head(x) # B,T,vocab_size
    B,T,C = logits.shape
    
    if target is not None:
      logits = logits.view(B*T,C)
      target = target.view(B*T,)
      loss = F.cross_entropy(logits, target)
    else:
      loss = None

    return logits, loss

  def generate(self, idx, max_num_tokens):
    for _ in range(max_num_tokens):
      logits, loss = self(idx) # B x T x C 
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

        



eval_iters = 200
batch_size = 32
eval_interval = 300
max_steps = 10000
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Detected Device", device)
m = BigramLanguageModel(vocab_size, n_embd).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(max_steps):
  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if step % eval_interval == 0:
      out_dict = evaluate_model(m)
      print(f"Step {step}/{max_steps}: Train Loss {out_dict['train_loss']:.4f}, Val Loss {out_dict['val_loss']:.4f}")

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_num_tokens=500)[0].tolist()))