import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

# utils
def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 1, 2).contiguous()
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 1, 2)

def merge_heads(x):
    return merge_states(torch.transpose(x, 1, 2))

def split_states(x, n):
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + (n, m // n)
    return torch.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + (np.prod(x_shape[-2:]),)
    return torch.reshape(x, new_x_shape)

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.transpose(a, 1, 2).contiguous()
        a = torch.reshape(a, [n, t, embd])
    return a

class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx, self.blocksize)

# Define the memory-enhanced transformer with sparse attention
class MemoryEnhancedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, compressive_memory_size, attn_mode, local_attn_ctx=None, blocksize=32):
        super(MemoryEnhancedTransformer, self).__init__()
        self.compress_memory = nn.Linear(input_dim, compressive_memory_size)
        self.sparse_attention_blocks = nn.ModuleList([
            SparseAttention(heads=num_heads, attn_mode=attn_mode, local_attn_ctx=local_attn_ctx, blocksize=blocksize)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(compressive_memory_size, 1)

    def forward(self, query, historical_stream):
        # Compress historical stream into memory
        compressive_memory = self.compress_memory(historical_stream)
        
        # Process each session with sparse attention
        for block in self.sparse_attention_blocks:
            compressive_memory = block(query, compressive_memory, compressive_memory)
        
        # Use query for QKV calculation
        query_memory = torch.cat((query, compressive_memory), dim=1)
        saliency_representation = self.output_layer(query_memory)
        
        return saliency_representation
    

# --------------------------------------Dataset Loader-----------------------------------------
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class LongMemEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sessions = self.data[idx]["haystack_sessions"]
        session_ids = self.data[idx]["haystack_session_ids"]
        query_list = []
        historical_stream = []
        has_answer = []

        # Tokenize the query using AutoTokenizer
        query_tokens = tokenizer.encode(self.data[idx]["question"], add_special_tokens=True)
        query_ = torch.randn(len(query_tokens), n_embd)

        for session in sessions:
            session_content = []
            for turn in session:
                session_content.append(turn["content"])
            # Tokenize the session content using AutoTokenizer
            content_tokens = tokenizer.encode("".join(session_content), add_special_tokens=True)
            encoded_content = torch.randn(min(len(content_tokens), MAX_TOKEN_LENGTH), n_embd)
            historical_stream.append(encoded_content)
            query_list.append(query_)

        for id, encoded_content in zip(session_ids, historical_stream):
            has_answer_value = torch.randn(1, n_embd)
            has_answer.append(has_answer_value)

        # Stack queries, historical_streams and has_answers
        query = torch.stack([F.pad(q, (0, 0, 0, MAX_TOKEN_LENGTH - q.size(0))) for q in query_list], dim=0)
        historical_stream = torch.stack([F.pad(h, (0, 0, 0, MAX_TOKEN_LENGTH - h.size(0))) for h in historical_stream], dim=0)
        has_answer = torch.stack(has_answer, dim=0).expand(-1, MAX_TOKEN_LENGTH, -1)
        return query, historical_stream, has_answer

class FlattenedLongMemEvalDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for item in tqdm(data, desc="Loading Dataset"):
            sessions = item["haystack_sessions"]
            session_ids = item["haystack_session_ids"]

            # Tokenize the query using AutoTokenizer
            query_tokens = tokenizer.encode(item["question"], add_special_tokens=True)
            query = torch.randn(min(len(query_tokens), MAX_QUERY_LENGTH), n_embd)
            query = F.pad(query, (0, 0, 0, MAX_QUERY_LENGTH - query.size(0)))

            for session, session_id in zip(sessions, session_ids):
                session_content = "".join(turn["content"] for turn in session)
                # Tokenize the session content using AutoTokenizer
                content_tokens = tokenizer.encode(session_content, add_special_tokens=True)
                encoded_content = torch.randn(min(len(content_tokens), MAX_TOKEN_LENGTH), n_embd)
                encoded_content = F.pad(encoded_content, (0, 0, 0, MAX_TOKEN_LENGTH - encoded_content.size(0)))

                # Create has_answer with shape (1, n_embd)
                has_answer_value = torch.randn(1, n_embd)
                has_answer_value = has_answer_value.expand(MAX_TOKEN_LENGTH, n_embd)

                self.samples.append((query, encoded_content, has_answer_value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    queries, historical_streams, labels = zip(*batch)
    queries = torch.stack(queries)
    historical_streams = torch.stack(historical_streams)
    labels = torch.stack(labels)
    return queries, historical_streams, labels

dataset = FlattenedLongMemEvalDataset(longmemeval_s)


def train_model(model, dataset, epochs=3, batch_size=16, learning_rate=1e-4):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print("loading data is done")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                if batch is None:
                    continue
                query, historical_stream, has_answer = batch
                query, historical_stream, has_answer = (
                    query.to(device).float(),
                    historical_stream.to(device).float(),
                    has_answer.to(device),
                )
                optimizer.zero_grad()
                
                # Ensure the historical_stream is reshaped into 3D tensor
                historical_stream = historical_stream.view(historical_stream.size(0), -1, n_embd)
                
                # Correct dimension mismatch for the flattening operation
                historical_stream_flattened = historical_stream.view(-1, n_embd)
                compressive_memory = model.compress_memory(historical_stream_flattened)
                hidden_representations = compressive_memory.view(historical_stream.size(0), historical_stream.size(1), -1)

                # Process each session with sparse attention
                for block in model.sparse_attention_blocks:
                    hidden_representations = block(query, hidden_representations, hidden_representations)
                    
                # Use query for QKV calculation
                query_memory = torch.cat((query, hidden_representations), dim=2)  # Change dim=1 to dim=2
                query_memory_reshaped = query_memory.view(batch_size, -1, compressive_memory_size)
                saliency_representation = model.output_layer(query_memory_reshaped)
                
                # Convert hidden representations to a saliency map
                saliency_map = torch.sigmoid(saliency_representation.view(-1))
                # Calculate loss
                loss = criterion(saliency_map, has_answer.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
            pbar.update(1)


# Initialize MemoryEnhancedTransformer
n_embd = 256
heads = 4
attn_mode = "local"
local_attn_ctx = 32
blocksize = 32
input_dim = n_embd
hidden_dim = n_embd * 2
num_layers = 2
compressive_memory_size = n_embd
transformer_model = MemoryEnhancedTransformer(
    input_dim,
    hidden_dim,
    heads,
    num_layers,
    compressive_memory_size,
    attn_mode,
    local_attn_ctx,
    blocksize,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use only one GPU
model = transformer_model
model.to(device)

# Use the LongMemEvalDataset for training
# dataset = FlattenedLongMemEvalDataset(longmemeval_s)
train_model(model, dataset)