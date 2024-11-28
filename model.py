import torch
import torch.nn as nn
import math

##The backbone model can use any neural network,
# here we take transformer and GPT as examples
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_length, embedding_dim]
        """
        return x + self.pe[:x.size(1)].transpose(0, 1)
class CustomTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, predict_length, nhead=2, num_layers=4, dropout=0.1,
                 freeze_parameters=True):
        super(CustomTransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(hidden_dim, predict_length)

        if freeze_parameters:
            for i in range(num_layers - 2):
                for param in self.transformer_encoder.layers[i].parameters():
                    param.requires_grad = False

    def forward(self, x, output_type=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        mask = mask.to(x.device)

        features = self.transformer_encoder(x, mask=mask)

        if output_type == 'short_term':
            return features[:, -1, :]
        elif output_type == 'long_term':
            return features[:, -1, :]
        else:
            return self.output_projection(features[:, -1, :])

class GPT2Model(nn.Module):
    def __init__(self, gpt2_model, input_dim, hidden_dim):
        super(GPT2Model, self).__init__()
        self.gpt2 = gpt2_model
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_long_term = nn.Linear(self.gpt2.config.n_embd, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, predict_length)
        # 冻结大部分 GPT-2 层
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # 解冻最后两层和输出层
        for param in self.gpt2.transformer.h[-5:].parameters():
            param.requires_grad = True
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True
    def forward(self, x, output_type=None):
        x = self.linear_in(x)
        batch_size, seq_len, hidden_size = x.size()
        if seq_len < self.gpt2.config.n_positions:
            padding = self.gpt2.config.n_positions - seq_len
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
        outputs = self.gpt2.transformer(inputs_embeds=x, output_hidden_states=True)
        if output_type == 'short_term':
            return outputs.hidden_states[-2][:, -1, :]
        elif output_type == 'long_term':
            long_term_features = self.linear_long_term(outputs.hidden_states[-2][:, -1, :])
            return long_term_features
        else:
            final_features = outputs.hidden_states[-1][:, -1, :]
            return self.linear_out(final_features)