# banking77: prima e dopo LoRA

Esperimento di fine-tuning LoRA (QLoRA, modello a 4 bit) di `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
sul dataset [mteb/banking77](https://huggingface.co/datasets/mteb/banking77): 77 intenti bancari.
Inferenza e training con `mlx-lm`, decoding greedy. Il modello viene letto da `~/.omlx/models` se presente.

## Flusso

```bash
uv run scripts/download.py                     # 1. dataset → data/ (train 8994, valid 999, test 3076)

uv run scripts/evaluate.py --name base-full --prompt full          # 2. baseline

uv run scripts/finetune.py                     # 3. LoRA → adapters/banking77/

uv run scripts/evaluate.py --name ft-full    --prompt full    --adapter adapters/banking77   # 4.
uv run scripts/evaluate.py --name ft-minimal --prompt minimal --adapter adapters/banking77

uv run scripts/compare.py base-full ft-full ft-minimal             # 5. tabella prima/dopo
```

Per iterare velocemente aggiungi `--limit 200` alle valutazioni e `--iters 300` al fine-tuning.
Tutti gli script hanno `--help`. La guida passo per passo è in [GUIDA.md](GUIDA.md).

## Scenari

| Nome | Prompt | Cosa misura |
|---|---|---|
| `base-full` | system prompt con le 77 etichette | il modello base, che senza la lista non conoscerebbe le classi |
| `ft-full` | stesso prompt della baseline | confronto alla pari |
| `ft-minimal` | solo istruzione breve (lo stesso usato in training) | il guadagno reale: niente lista, molti meno token |

## Metriche

- **Formattate correttamente**: la risposta, ripulita da spazi, virgolette e punto finale, è una delle 77 etichette.
- **Corrette**: formattata e uguale all'etichetta attesa.
- **Token**: media per richiesta, separata in prompt e generati.
- **Latenza**: tempo medio per richiesta, prompt più generazione.

Le singole predizioni sono in `results/<nome>.jsonl`, le metriche aggregate in `results/<nome>.summary.json`.

## Note

- Il validation set usato durante il training è ricavato dal train: il test non viene mai visto.
- Di default il training usa il prompt minimo. `download.py --train-prompt full` genera esempi con la lista
  delle etichette (training circa 10 volte più lento).
