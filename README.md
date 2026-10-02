# Fine-tuning LoRA di modelli locali

Script per misurare un modello installato in oMLX, addestrarlo con
LoRA su un dataset di classificazione scaricato da Hugging Face e misurarlo di nuovo. Inferenza e
training usano `mlx-lm` in processo, con decoding greedy. Ogni dataset scaricato ha un nome locale,
quindi se ne possono tenere più di uno.

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

# 4. misurare il modello addestrato
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77 \
              --prompt-type minimal --output results/ft-minimal.jsonl
```

Tutti i comandi hanno `--help` e si possono lanciare da qualsiasi cartella. Per una prova veloce
aggiungi `--limit 200` alle valutazioni e usa `--config configs/smoke.yaml` per il training.
La guida passo per passo è in [GUIDA.md](GUIDA.md).

## I comandi

| Comando | Argomenti obbligatori | Opzioni utili |
|---|---|---|
| `download.sh` | `--source` (id Hugging Face), `--name` (nome locale) | `--subset`, `--text-column`, `--label-column`, `--valid-ratio`, `--test-ratio`, `--overwrite` |
| `evaluate.sh` | `--model`, `--dataset-name`, `--prompt-type full\|minimal`, `--output` | `--adapter`, `--limit`, `--split valid` |
| `fine-tune.sh` | `--model`, `--dataset-name`, `--config`, `--output` | `--overwrite` |

**`--model`** è il nome di un modello installato in oMLX: quello della cartella
(`Qwen2.5-0.5B-Instruct-4bit`) o il percorso completo (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`).
Le cartelle si leggono da `~/.omlx/settings.json`. Il server oMLX non deve essere attivo; se lo è,
gli script lo segnalano, perché il modello occuperebbe memoria due volte.

**`download.sh`** riconosce da solo le colonne di testo e di etichetta, le mostra a schermo e
ricava dal train gli split mancanti (prima il test, poi il validation). Se la scelta delle
colonne è sbagliata si corregge con `--text-column` e `--label-column`.

**`--prompt-type`** sceglie il system prompt: `full` contiene l'elenco delle etichette, `minimal`
solo un'istruzione breve. I due testi sono in `datasets/<nome>/prompts.yaml`, generati al download
con una formulazione neutra da adattare al dominio.

**`--adapter`** carica un adapter prodotto da `fine-tune.sh`. Lo script controlla nel suo
`manifest.json` che sia stato addestrato sullo stesso modello, e avvisa se è stato addestrato su
un altro dataset.

## Il config del training

Il file passato a `--config` sceglie il prompt degli esempi di training e la durata. Le epoche
diventano iterazioni in base alla dimensione del train (`epochs × esempi / batch_size`). Esempio
completo e commentato in [`configs/lora-base.yaml`](configs/lora-base.yaml):

```yaml
prompt_type: minimal      # prompt degli esempi di training: minimal | full
epochs: 1                 # oppure iters: <n>, non entrambi
batch_size: 8
learning_rate: 1.0e-4
num_layers: 16
lora: {rank: 8, scale: 20.0, dropout: 0.0}
max_seq_length: 1024
mlx_lm: {}                # opzioni di mlx_lm lora passate così come sono
```

Le chiavi sconosciute vengono rifiutate, così un errore di battitura non passa inosservato.

## Struttura delle cartelle

```
datasets/<nome>/
  manifest.json          sorgente, colonne scelte, split e loro origine, seed
  labels.json            le etichette
  prompts.yaml           i prompt minimal e full, modificabili
  raw/{train,valid,test}.jsonl
adapters/<nome>/
  manifest.json          modello, dataset, prompt e config usati
  train_config.yaml      configurazione completa passata a mlx_lm lora
  data/{train,valid}.jsonl   esempi in formato chat
  adapters.safetensors   l'adapter finale, più i checkpoint intermedi
configs/                 config di training
results/                 output delle valutazioni (consigliato, non obbligatorio)
```

`evaluate.sh` scrive le predizioni nel file `--output` e le metriche accanto, con suffisso
`.summary.json` al posto di `.jsonl`. `scripts/compare.py` legge i riepiloghi che stanno
direttamente in `results/`.

## Metriche

- **Formattate correttamente**: la risposta, ripulita da spazi, virgolette e punto finale e portata
  in minuscolo, è una delle etichette.
- **Corrette**: formattata e uguale all'etichetta attesa, normalizzata allo stesso modo.
- **Token**: media per richiesta, separata in prompt e generati.
- **Latenza**: tempo medio per richiesta, prompt più generazione.

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
