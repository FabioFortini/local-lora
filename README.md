# Fine-tuning LoRA di modelli locali

Script per misurare un modello installato in oMLX, addestrarlo con LoRA su un dataset di
classificazione di Hugging Face e misurarlo di nuovo. Inferenza e training usano `mlx-lm` in
processo, con decoding greedy. Ogni dataset scaricato ha un nome locale, quindi se ne possono tenere
più di uno.

## Flusso

```bash
# 1. scaricare un dataset e dargli un nome locale
./download.sh --source mteb/banking77 --name banking77

# 2. misurare il modello base
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --prompt-type full --output results/base-full.jsonl

# 3. addestrare un adapter LoRA
./fine-tune.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
               --config configs/lora-base.yaml --output adapters/qwen05-banking77

# 4. misurare il modello addestrato, con lo stesso prompt della baseline e con quello breve
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77 \
              --prompt-type full --output results/ft-full.jsonl
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77 \
              --prompt-type minimal --output results/ft-minimal.jsonl

# 5. confrontare
uv run scripts/compare.py base-full ft-full ft-minimal
```

Tutti i comandi hanno `--help` e si possono lanciare da qualsiasi cartella. Per iterare in fretta
aggiungi `--limit 200` alle valutazioni e usa un config con `iters: 300`.

## I comandi

| Comando | Argomenti obbligatori | Opzioni utili |
|---|---|---|
| `download.sh` | `--source` (id Hugging Face), `--name` (nome locale) | `--subset`, `--text-column`, `--label-column`, `--valid-ratio`, `--test-ratio`, `--overwrite` |
| `evaluate.sh` | `--model`, `--dataset-name`, `--prompt-type full\|minimal`, `--output` | `--adapter`, `--limit`, `--split valid` |
| `fine-tune.sh` | `--model`, `--dataset-name`, `--config`, `--output` | `--overwrite` |

**`--model`** è il nome di un modello installato in oMLX: quello della cartella
(`Qwen2.5-0.5B-Instruct-4bit`) o il percorso completo (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
Le cartelle si leggono da `~/.omlx/settings.json`; se il nome non viene trovato, lo script elenca i
modelli installati. Il server oMLX non deve essere attivo, e se lo è gli script lo segnalano, perché
il modello occuperebbe memoria due volte.

**`download.sh`** riconosce da solo le colonne di testo e di etichetta e le mostra a schermo: se
sono sbagliate si correggono con `--text-column` e `--label-column`. Gli split mancanti si ricavano
dal train, prima il test e poi il validation, quindi il test non viene mai visto in training. Con lo
stesso `--seed` il comando rigenera file identici.

**`--prompt-type`** sceglie il system prompt: `full` contiene l'elenco delle etichette, `minimal`
solo un'istruzione breve. I testi stanno in `datasets/<nome>/prompts.yaml`.

## I prompt di un dataset

Dopo aver scaricato un dataset nuovo, scrivi un `prompts.yaml` adatto al suo dominio prima di
misurare la baseline. `download.sh` genera due prompt neutri, uguali per ogni dataset («Classify the
text into its label»), mentre un prompt che dice al modello di cosa si parla di solito migliora la
baseline. Per banking77, per esempio:

```yaml
# {labels} is replaced with the label list, one per line. download.sh keeps edits unless --overwrite.
minimal: >-
  Classify the banking customer message into its intent label.
  Answer with the label only.
full: |-
  You are a banking customer support classifier. Classify the customer message into exactly one of the following intent labels:
  {labels}

  Answer with the label only, exactly as written above, with no other text.
```

Il prompt va deciso una volta e poi lasciato fermo: baseline, training e valutazioni sono
confrontabili solo se usano lo stesso testo, e un adapter va valutato con il prompt con cui è stato
addestrato. Per questo `prompts.yaml` è l'unico file di `datasets/` tracciato in git: chi clona il
repository e scarica lo stesso dataset ottiene gli stessi dati e trova già i prompt giusti, che
`download.sh` non sovrascrive.

**`--adapter`** carica un adapter prodotto da `fine-tune.sh`, dopo aver controllato nel suo
`manifest.json` che sia stato addestrato sullo stesso modello.

## Il config del training

[`configs/lora-base.yaml`](configs/lora-base.yaml) contiene i valori di partenza. Le epoche
diventano iterazioni in base alla dimensione del train (`epochs × esempi / batch_size`), e le chiavi
sconosciute vengono rifiutate.

| Parametro | Default | Effetto |
|---|---|---|
| `prompt_type` | minimal | prompt degli esempi di training; `full` rende il training circa 10 volte più lento |
| `epochs` / `iters` | 1 epoca | più passate sui dati, ma più rischio di overfitting |
| `batch_size` | 8 | esempi per iterazione |
| `learning_rate` | 1.0e-4 | se la loss oscilla, abbassalo (scritto con il punto, `1.0e-4`) |
| `lora.rank` | 8 | capacità dell'adapter |
| `num_layers` | 16 | quanti layer adattare (`-1` = tutti) |
| `mlx_lm` | `{}` | opzioni passate così come sono a `mlx_lm lora`, per esempio `grad_checkpoint: true` |

Nel log del training contano due righe: `Train loss` ogni `steps_per_report` iterazioni e `Val loss`
ogni `steps_per_eval`. Se la val loss scende e si stabilizza va tutto bene; se la train loss continua
a scendere mentre la val loss risale, è overfitting e conviene ridurre le epoche. Le righe
`Calculating loss...` sono la barra di avanzamento della validazione e si possono ignorare.

## Struttura delle cartelle

```
datasets/<nome>/          manifest.json, labels.json, prompts.yaml, raw/{train,valid,test}.jsonl
adapters/<nome>/          manifest.json, train_config.yaml, data/, adapters.safetensors e checkpoint
configs/                  config di training
results/                  output delle valutazioni
```

`datasets/`, `adapters/` e `results/` non sono in git e si rigenerano con i comandi, tranne i
`prompts.yaml` dei dataset, che sono tracciati.
`evaluate.sh` scrive le predizioni nel file `--output` e le metriche accanto, con `.summary.json` al
posto di `.jsonl`. `compare.py` legge i riepiloghi che stanno direttamente in `results/`.

## Metriche

| Metrica | Significato |
|---|---|
| Formattate correttamente | la risposta, ripulita da spazi, virgolette e punto finale e portata in minuscolo, è una delle etichette |
| Corrette | formattata e uguale all'etichetta attesa, normalizzata allo stesso modo |
| Token di prompt e generati | media per richiesta |
| Latenza | tempo medio per richiesta, prompt più generazione |

## Test

```bash
uv run pytest
```

## Limiti noti

Il progetto per ora non gestisce i casi seguenti nel codice: vanno tenuti presenti quando si
sceglie un modello o un dataset.

**Compiti.** È supportata solo la classificazione a etichetta singola. Multi-etichetta, domande e
risposte, riassunti e generazione libera richiedono metriche e prompt diversi.

**Dataset.**
- Le etichette devono avere un nome: una colonna intera senza nomi (né `ClassLabel` né una
  colonna `<x>_text` accanto) viene rifiutata.
- Oltre le 200 etichette il prompt `full` diventa molto lungo e può superare `max_seq_length`:
  con 77 etichette costa già circa 450 token di prompt per richiesta, contro 43 con `minimal`.
- I dataset che dipendono da uno script di caricamento non si scaricano con `datasets` 5.x.
- I dataset privati o con accesso controllato richiedono un token: `export HF_TOKEN=...`.
- Le righe senza testo o senza etichetta vengono scartate e contate nel manifest. Lo split di
  test di GLUE, per esempio, non ha etichette e viene quindi ricavato dal train.

**Modelli.**
- Servono pesi in formato MLX (safetensors, tipicamente `mlx-community`). I modelli GGUF di Ollama
  o LM Studio non si possono addestrare.
- L'architettura deve essere supportata da `mlx-lm` (Llama, Qwen 2/2.5/3, Gemma, Mistral, Phi e
  altre).
- Su un modello quantizzato si può fare solo LoRA (QLoRA); il full fine-tuning richiede i pesi non
  quantizzati.
- Il modello deve avere un chat template: i modelli base che non lo hanno falliscono in
  valutazione e in training.
- Alcuni template, per esempio nella famiglia Gemma, non accettano il ruolo `system` e falliscono.
- I modelli che ragionano prima di rispondere (Qwen3 e simili) producono un blocco `<think>` che
  con `--max-tokens 32` viene troncato: la risposta risulta non formattata e l'accuratezza è
  falsamente vicina a zero.

**Memoria (36 GB, Apple M4 Max).** Il training su Qwen2.5-0.5B a 4 bit ha un picco di 3,8 GB.
Come ordine di grandezza, da verificare caso per caso: un 7-8B a 4 bit sta comodo, un 14B con
batch piccolo, un 30B a 4 bit solo con `grad_checkpoint: true` sotto `mlx_lm` e batch 1-2, un 70B
non ci sta. Se il server oMLX ha il modello in memoria, la disponibilità si dimezza.

**Valutazione.** Avviene in processo, non attraverso il server oMLX, e quindi la latenza misurata
non include HTTP, scheduler e cache dei prefissi del server.
