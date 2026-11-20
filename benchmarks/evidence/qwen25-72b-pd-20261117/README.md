# Qwen2.5-72B-Instruct on Ascend910_9362

This is the BF16 AISimulate calibration for the SGLang
`0.5.20.dev388+ga6b542813` runtime. One physical board corresponds to two
runtime NPUs. The recommended replay configuration uses 64 physical boards
across eight hosts, with 11 TP8 Prefill replicas, 5 TP8 Decode replicas, and a
16,384-token Prefill chunk.

## Replay

Run from this directory with the AISimulate build from this branch. Put the
pinned Qwen2.5-72B-Instruct model metadata from revision
`495f39366efef23836d0cfae4fbe635880d2be31` in `model_metadata/`. The
`config.json` SHA-256 must be
`14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29`.

Provide the prepared 32K AgentX trace as `agentx-32k.jsonl`, in
`agentic_mooncake` format. The run used 393 complete conversation trees
and the original inter-arrival timestamps. The published configuration
selects that benchmark's model, placement, and measured timing inputs.

```sh
aisimulate predict -c replay.json --stack engine --capture-per-request --output-dir replay-output
```

Keep each forward timing Parquet with its adjacent metadata JSON. The two
tables contain TP4 and TP8, Prefill and Decode measurements, rebased only
in the model path to match `replay.json`. The 16,384-token chunk table is
selected by the example. The 8,192-token table supports separate chunk
comparisons with matching scheduler settings.

[The transfer profile](systems/ascend9362_qwen72_pd.json) records the
median same-host and cross-host KV transfer curves and the replica placement
for this TP8 configuration. [The mixed-TP validation](mixed-sender-queue-native-validation-r2.json)
records the timing checks for the per-destination queue.
