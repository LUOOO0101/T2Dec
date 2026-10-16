import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train',
        embedding_batch_size = 8
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.embedding_batch_size = embedding_batch_size
        self.len = self.input_len-1
        self.is_web_traffic = self.data_path in ['WikiWeb', 'WikiPopular'] or self.data_path.startswith('KaggleWebTraffic')

        self.page_texts = []
        if self.data_path == 'WikiPopular':
            text_path = Path(__file__).resolve().parents[1] / 'datasets' / 'traffic_text_sampled.xlsx'
            text_table = pd.read_excel(text_path)
            self.page_texts = text_table['text'].fillna('').astype(str).tolist()
        
        # Embeddings are generated from the already cached checkpoint so runs remain reproducible offline.
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name, local_files_only=True).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        if self.is_web_traffic:
            values_str = ", ".join([f"{value:.3f}" for value in values])
        else:
            values_str = ", ".join([str(int(value)) for value in values])

        # Last token
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))#其实就是最后一个元素减掉第一个元素
        trends_str = f"{trends.item():.3f}" if self.is_web_traffic else f"{trends.item():0f}"
        
        # Date
        if self.data_path in ['FRED', 'ILI'] or self.is_web_traffic:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else: # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        # print("in_prompt: ", in_prompt)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings#保存最后一层隐藏层

    def generate_embeddings(self, in_data, in_data_mark):
        input_templates = {
                'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            'WikiWeb': "From [t1] to [t2], the Wikipedia pageview counts were value1, ..., valuen every day. The total trend value was Trends",
            'WikiPopular': "From [t1] to [t2], the Wikipedia pageview counts were value1, ..., valuen every day. The total trend value was Trends. Page context: PageText",
            'KaggleWebTrafficWeekly': "From [t1] to [t2], the Wikipedia pageview counts were value1, ..., valuen every week. The total trend value was Trends",
            'KaggleWebTrafficDaily': "From [t1] to [t2], the Wikipedia pageview counts were value1, ..., valuen every day. The total trend value was Trends",
                'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
            }

        input_template = input_templates.get(self.data_path)
        if input_template is None and self.data_path.startswith('KaggleWebTraffic'):
            input_template = "From [t1] to [t2], the Wikipedia pageview counts were value1, ..., valuen every day. The total trend value was Trends"
        elif input_template is None:
            input_template = input_templates['FRED']

        tokenized_prompts = []
        max_token_count = 0
        for i in range(len(in_data)):
            for j in range(in_data.shape[2]):  # 遍历每个通道（变量）
                template = input_template
                if self.data_path == 'WikiPopular' and j < len(self.page_texts):
                    # Keep the static context bounded so numeric history remains in GPT-2's context window.
                    text = ' '.join(self.page_texts[j].split()[:120])
                    template = template.replace('PageText', text)
                tokenized_prompt = self._prepare_prompt(template, in_data, in_data_mark, i, j).to(self.device)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

        # Batch prompts from the same window to avoid one GPT-2 call per channel.
        last_token_emb = torch.zeros(
            (len(in_data), self.d_model, in_data.shape[2]),
            dtype=torch.float32,
            device=self.device,
        )
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(tokenized_prompts), self.embedding_batch_size):
            batch = tokenized_prompts[start : start + self.embedding_batch_size]
            lengths = [prompt.shape[1] for _, prompt, _ in batch]
            input_ids = torch.full(
                (len(batch), max(lengths)), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (_, prompt, _) in enumerate(batch):
                length = prompt.shape[1]
                input_ids[row, :length] = prompt[0]
                attention_mask[row, :length] = 1

            with torch.no_grad():
                prompt_embeddings = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
            for row, (i, _, j) in enumerate(batch):
                last_token_emb[i, :, j] = prompt_embeddings[row, lengths[row] - 1, :]

        return last_token_emb # output [batch, d_model, channel]
