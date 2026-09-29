# OSpec ensemble adaptation

Start with `.venv/bin/python -m methods.run ospec --chunk-size 5
--ensemble-lrs 1e-5,2e-5,3e-5 --epsilon 0.1`. After five observed completed
requests, three CPU learners train on the captured verification records;
cumulative-KL softmax weights combine trainable draft parameters for the
next chunk. Inference still uses the identical GPU vLLM EAGLE-3 path. This
adapts the chunk/ensemble policy, not the original OnlineSPEC pipeline;
see [method limitations](../README.md).
