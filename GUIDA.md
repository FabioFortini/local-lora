# Guida all'esperimento completo

Come eseguire un esperimento dall'inizio alla fine: dataset, baseline, fine-tuning LoRA, nuova
valutazione e confronto. L'esempio usa banking77 (77 intenti bancari) e
`Qwen2.5-0.5B-Instruct-4bit`; per un altro dataset o un altro modello cambiano solo i nomi.
Ogni comando ha `--help`.

## 1. Scaricare il dataset

```bash
./download.sh --source mteb/banking77 --name banking77
```

Lo script mostra le colonne scelte (qui `text` e `label_text`) e scrive in `datasets/banking77/`:

- **`raw/`**: train (8.994), valid (999) e test (3.076), con testo ed etichetta.
- **`labels.json`**: le 77 etichette.
- **`prompts.yaml`**: i prompt `minimal` e `full`. Per banking77 contiene i prompt originali
  dell'esperimento, specifici per il dominio bancario.
- **`manifest.json`**: sorgente, colonne, numero di esempi e origine di ogni split.

banking77 non ha uno split di validation, quindi viene ricavato dal train: il test non viene mai
visto durante il training. Rilanciando il comando con lo stesso `--seed` si ottengono file identici,
e `prompts.yaml` non viene toccato.

### Con un dataset diverso

```bash
./download.sh --source dair-ai/emotion --name emotion
./download.sh --source nyu-mll/glue --subset sst2 --name sst2
```

Controlla la tabella delle colonne stampata a schermo. Se il testo o l'etichetta sono sbagliati,
rilancia con `--text-column` o `--label-column`. Poi apri `datasets/<nome>/prompts.yaml` e adatta
i due prompt al dominio: quelli generati sono neutri («Classify the text into its label»), e un
prompt che dice al modello di cosa si parla di solito migliora la baseline.

## 2. Misurare la baseline

```bash
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --prompt-type full --output results/base-full.jsonl
```

Senza `--limit` valuta tutti i 3.076 esempi di test (circa 6 minuti). La barra di avanzamento
mostra l'accuratezza parziale; alla fine compaiono le metriche e gli errori più frequenti, e
vengono salvati:

- **`results/base-full.jsonl`**: ogni singola risposta del modello, utile per capire dove sbaglia.
- **`results/base-full.summary.json`**: le metriche aggregate e i metadati della valutazione
  (modello, dataset, adapter, prompt, split), letti da `compare.py`.

Se rilanci con lo stesso `--output`, i file vengono sovrascritti.

## 3. Fare il fine-tuning

```bash
./fine-tune.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
               --config configs/lora-base.yaml --output adapters/qwen05-banking77 2>&1 | tee finetune.log
```

Il `tee` è facoltativo: salva il log per rileggerlo dopo. Con `configs/lora-base.yaml` il training
fa un'epoca con batch 8, cioè 1.125 iterazioni sui 8.994 esempi. Nel log ci sono due tipi di riga
da guardare:

```
Iter 50:  Train loss 1.450, ...     # ogni steps_per_report iterazioni
Iter 200: Val loss 0.533, ...       # ogni steps_per_eval, calcolata sul validation
```

- **Tutto bene:** la val loss scende e poi si stabilizza.
- **Overfitting:** la train loss continua a scendere mentre la val loss risale. Riduci `epochs`.
- **Molte righe "Calculating loss...":** sono la barra di avanzamento della validazione di
  mlx-lm, si possono ignorare.

L'adapter finale finisce in `adapters/qwen05-banking77/adapters.safetensors`, con i checkpoint
intermedi ogni `save_every` iterazioni. Nella stessa cartella ci sono gli esempi in formato chat
usati per il training, il `train_config.yaml` passato a mlx-lm e il `manifest.json`. Se la cartella
contiene già un adapter, lo script si ferma: scegli un'altra cartella o aggiungi `--overwrite`.

## 4. Valutare il modello addestrato

```bash
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77 \
              --prompt-type full --output results/ft-full.jsonl
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77 \
              --prompt-type minimal --output results/ft-minimal.jsonl
```

- **`ft-full`** usa lo stesso prompt della baseline, con l'elenco delle etichette: è il confronto
  alla pari.
- **`ft-minimal`** usa il prompt breve, lo stesso del training: mostra quanti token si risparmiano.

La seconda valutazione è più veloce della prima, perché il prompt è circa 10 volte più corto.

## 5. Confrontare

```bash
uv run scripts/compare.py base-full ft-full ft-minimal
```

La prima colonna è la baseline. Per ogni altro scenario c'è una colonna Δ con la differenza, in
verde se migliora e in rosso se peggiora. `compare.py` cerca i riepiloghi direttamente in
`results/`: per usarlo, salva lì le valutazioni.

## Metriche

| Metrica | Significato |
|---|---|
| Formattate correttamente | la risposta, ripulita da spazi, virgolette e punto finale e portata in minuscolo, è una delle etichette |
| Corrette | formattata e uguale all'etichetta attesa, normalizzata allo stesso modo |
| Token di prompt | media per richiesta dei token in input |
| Token generati | media per richiesta dei token prodotti dal modello |
| Latenza | tempo medio per richiesta, prompt più generazione |

## Provare varianti

Ogni variante è un config diverso, salvato in una cartella di adapter diversa e valutato con un
`--output` diverso, così nessun risultato sovrascrive gli altri:

```bash
cp configs/lora-base.yaml configs/r16-2ep.yaml     # poi imposta lora.rank: 16 e epochs: 2
./fine-tune.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
               --config configs/r16-2ep.yaml --output adapters/qwen05-banking77-r16-2ep
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/qwen05-banking77-r16-2ep \
              --prompt-type minimal --output results/r16-2ep-minimal.jsonl
uv run scripts/compare.py base-full ft-minimal r16-2ep-minimal
```

Parametri principali del config:

| Parametro | In `lora-base.yaml` | Effetto |
|---|---|---|
| `epochs` | 1 | più passate sui dati, ma anche più rischio di overfitting; in alternativa `iters` |
| `batch_size` | 8 | esempi per iterazione |
| `lora.rank` | 8 | capacità dell'adapter: più alto significa più parametri addestrabili |
| `learning_rate` | 1.0e-4 | se la loss oscilla, abbassalo (scrivilo con il punto: `1.0e-4`) |
| `num_layers` | 16 | quanti layer adattare (`-1` = tutti) |
| `prompt_type` | minimal | prompt usato negli esempi di training |

Prima di lanciare una variante lunga, provala in piccolo: `configs/smoke.yaml` fa 30 iterazioni,
e `--limit 300` riduce la valutazione. Se non promette nulla su quel campione, difficilmente
cambierà sul test completo.

## Allenare con il prompt completo

Con `prompt_type: minimal` nello scenario `ft-full` il modello riceve un prompt diverso da quello
visto in training. Per allenare con l'elenco delle etichette nel prompt basta un config con
`prompt_type: full`: non serve riscaricare il dataset. Il training diventa circa 10 volte più
lento, perché ogni esempio è circa 10 volte più lungo.

## Cambiare modello

Il modello deve essere installato in oMLX, cioè stare in una delle cartelle elencate in
`~/.omlx/settings.json` (di default `~/.omlx/models`). Se il nome non viene trovato, lo script
elenca i modelli installati. Prima di scegliere un modello grande leggi i limiti di memoria nel
[README](README.md#limiti-noti), e ferma il server oMLX (`omlx stop`) se ha già caricato modelli.

## LoRA e full fine-tuning

Questi script usano LoRA: i pesi del modello restano fermi e si allena un piccolo adapter (con
Qwen2.5-0.5B e i parametri di default, circa lo 0,6% dei parametri). Su un modello quantizzato si
tratta tecnicamente di QLoRA. Il full fine-tuning, che aggiorna tutti i pesi, non si può fare su
un modello a 4 bit: servirebbe la versione non quantizzata, per esempio `Qwen/Qwen2.5-0.5B-Instruct`.

## L'adapter del primo esperimento

`adapters/banking77` è stato addestrato prima della versione generica degli script, con 1.200
iterazioni e il prompt minimal originale. Il suo `manifest.json` è stato ricostruito da
`train_config.yaml` e `finetune.log`, quindi si valuta come gli altri:

```bash
./evaluate.sh --model Qwen2.5-0.5B-Instruct-4bit --dataset-name banking77 \
              --adapter adapters/banking77 --prompt-type minimal --output results/ft-banking77-minimal.jsonl
```
