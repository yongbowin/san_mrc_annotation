# SAN Annotation

This project is a annotation for SAN, the raw official README id [here](README_official.md). This repo is based on [paper](https://arxiv.org/pdf/1809.09194.pdf).

## Model Arch

```
############# Model Arch of SAN #############
DNetwork(
  (dropout): DropoutWrapper()
  (lexicon_encoder): LexiconEncoder(
    (dropout): DropoutWrapper()
    (dropout_emb): DropoutWrapper()
    (dropout_cove): DropoutWrapper()
    (embedding): Embedding(90981, 300, padding_idx=0)
    (ContextualEmbed): ContextualEmbed(
      (embedding): Embedding(90981, 300, padding_idx=0)
      (rnn1): LSTM(300, 300, bidirectional=True)
      (rnn2): LSTM(600, 300, bidirectional=True)
    )
    (prealign): AttentionWrapper(
      (score_func): SimilarityWrapper(
        (score_func): DotProductProject(
          (dropout): DropoutWrapper()
          (proj_1): Linear(in_features=300, out_features=128, bias=False)
          (proj_2): Linear(in_features=300, out_features=128, bias=False)
        )
      )
    )
    (pos_embedding): Embedding(54, 12, padding_idx=0)
    (ner_embedding): Embedding(41, 8, padding_idx=0)
    (doc_pwnn): PositionwiseNN(
      (w_0): Conv1d(1224, 256, kernel_size=(1,), stride=(1,))
      (w_1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      (dropout): DropoutWrapper()
    )
    (que_pwnn): PositionwiseNN(
      (w_0): Conv1d(900, 256, kernel_size=(1,), stride=(1,))
      (w_1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      (dropout): DropoutWrapper()
    )
  )
  (doc_encoder_low): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(856, 128, bidirectional=True)
  )
  (doc_encoder_high): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(856, 128, bidirectional=True)
  )
  (query_encoder_low): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(856, 128, bidirectional=True)
  )
  (query_encoder_high): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(856, 128, bidirectional=True)
  )
  (query_understand): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(512, 128, bidirectional=True)
  )
  (deep_attn): DeepAttentionWrapper(
    (dropout): DropoutWrapper()
    (attn_list): ModuleList(
      (0): AttentionWrapper(
        (score_func): SimilarityWrapper(
          (score_func): DotProductProject(
            (dropout): DropoutWrapper()
            (proj_1): Linear(in_features=1412, out_features=128, bias=False)
            (proj_2): Linear(in_features=1412, out_features=128, bias=False)
          )
        )
      )
      (1): AttentionWrapper(
        (score_func): SimilarityWrapper(
          (score_func): DotProductProject(
            (dropout): DropoutWrapper()
            (proj_1): Linear(in_features=1412, out_features=128, bias=False)
            (proj_2): Linear(in_features=1412, out_features=128, bias=False)
          )
        )
      )
      (2): AttentionWrapper(
        (score_func): SimilarityWrapper(
          (score_func): DotProductProject(
            (dropout): DropoutWrapper()
            (proj_1): Linear(in_features=1412, out_features=128, bias=False)
            (proj_2): Linear(in_features=1412, out_features=128, bias=False)
          )
        )
      )
    )
  )
  (doc_understand): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(1280, 128, bidirectional=True)
  )
  (doc_self_attn): AttentionWrapper(
    (score_func): SimilarityWrapper(
      (score_func): DotProductProject(
        (dropout): DropoutWrapper()
        (proj_1): Linear(in_features=2436, out_features=128, bias=False)
        (proj_2): Linear(in_features=2436, out_features=128, bias=False)
      )
    )
  )
  (doc_mem_gen): OneLayerBRNN(
    (dropout): DropoutWrapper()
    (rnn): LSTM(512, 128, bidirectional=True)
  )
  (query_sum_attn): SelfAttnWrapper(
    (att): LinearSelfAttn(
      (linear): Linear(in_features=256, out_features=1, bias=True)
      (dropout): DropoutWrapper()
    )
  )
  (decoder): SAN(
    (attn_b): FlatSimilarityWrapper(
      (att_dropout): DropoutWrapper()
      (score_func): BilinearFlatSim(
        (linear): Linear(in_features=256, out_features=256, bias=True)
        (dropout): DropoutWrapper()
      )
    )
    (attn_e): FlatSimilarityWrapper(
      (att_dropout): DropoutWrapper()
      (score_func): BilinearFlatSim(
        (linear): Linear(in_features=256, out_features=256, bias=True)
        (dropout): DropoutWrapper()
      )
    )
    (rnn): GRUCell(256, 256)
    (dropout): DropoutWrapper()
  )
)
```

