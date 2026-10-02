# Guida all'esperimento completo

Come eseguire l'esperimento su tutti i dati: baseline, fine-tuning LoRA, nuova valutazione e confronto.
Tutti i comandi vanno lanciati dalla cartella del progetto. Ogni script ha `--help`.

## 1. Preparare i dati

```bash
uv run scripts/download.py
```

Scarica banking77 e scrive in `data/`:

- **`data/raw/`**: train (8.994), valid (999) e test (3.076) con testo ed etichetta, usati dalla valutazione.
- **`data/lora/`**: train e valid in formato chat, usati dal training.
- **`data/labels.json`**: le 77 etichette.

Il validation set è ricavato dal train, quindi il test non viene mai visto durante il training.
Va lanciato una volta sola: se lo rilanci con lo stesso `--seed` rigenera file identici.

## 2. Misurare la baseline

```bash
uv run scripts/evaluate.py --name base-full --prompt full
```

Senza `--limit` valuta tutti i 3.076 esempi di test (stima: circa 8 minuti). Durante l'esecuzione
la barra di avanzamento mostra l'accuratezza parziale. Alla fine stampa le metriche e gli errori più
frequenti, e salva:

- **`results/base-full.jsonl`**: ogni singola risposta del modello, utile per capire dove sbaglia.
- **`results/base-full.summary.json`**: le metriche aggregate, lette da `compare.py`.

Se rilanci con lo stesso `--name`, i file vengono sovrascritti.

## 3. Fare il fine-tuning

```bash
uv run scripts/finetune.py 2>&1 | tee finetune.log
```

Il `tee` è facoltativo: salva il log per rileggerlo dopo. Con i default fa 1.200 iterazioni con
batch 8, circa una passata completa sugli esempi di training. Nel log ci sono due tipi di riga da guardare:

```
Iter 50:  Train loss 1.450, ...     # ogni 50 iterazioni
Iter 200: Val loss 0.533, ...       # ogni 200, calcolata sui 999 esempi di validazione
```

- **Tutto bene:** la val loss scende e poi si stabilizza.
- **Overfitting:** la train loss continua a scendere mentre la val loss risale. Rilancia con meno iterazioni.
- **Molte righe "Calculating loss...":** sono la barra di avanzamento della validazione di mlx-lm, si possono ignorare.

L'adapter finale viene salvato in `adapters/banking77/adapters.safetensors`. Nella stessa cartella
ci sono anche `train_config.yaml` con i parametri usati e i checkpoint intermedi ogni 200 iterazioni.

## 4. Valutare il modello fine-tuned

```bash
uv run scripts/evaluate.py --name ft-full    --prompt full    --adapter adapters/banking77
uv run scripts/evaluate.py --name ft-minimal --prompt minimal --adapter adapters/banking77
```

- **`ft-full`** usa lo stesso prompt della baseline, con la lista delle etichette: è il confronto alla pari.
- **`ft-minimal`** usa il prompt breve, lo stesso del training: mostra quanti token si risparmiano.

La seconda valutazione è molto più veloce della prima, perché il prompt è circa 10 volte più corto.

## 5. Confrontare

```bash
uv run scripts/compare.py base-full ft-full ft-minimal
```

La prima colonna è la baseline. Per ogni altro scenario c'è una colonna Δ con la differenza,
in verde se migliora e in rosso se peggiora. Senza argomenti, `compare.py` mostra tutti i risultati
in `results/` nell'ordine in cui sono stati prodotti.

## Metriche

| Metrica | Significato |
|---|---|
| Formattate correttamente | la risposta, ripulita da spazi, virgolette e punto finale, è una delle 77 etichette |
| Corrette | formattata e uguale all'etichetta attesa |
| Token di prompt | media per richiesta dei token in input |
| Token generati | media per richiesta dei token prodotti dal modello |
| Latenza | tempo medio per richiesta, prompt più generazione |

## Provare varianti

Per confrontare più configurazioni dai a ogni training un `--adapter-path` diverso e a ogni
valutazione un `--name` diverso, così nessun risultato sovrascrive gli altri:

```bash
uv run scripts/finetune.py --rank 16 --iters 2000 --adapter-path adapters/r16-2000
uv run scripts/evaluate.py --name r16-minimal --prompt minimal --adapter adapters/r16-2000
uv run scripts/compare.py base-full ft-minimal r16-minimal
```

Parametri principali di `finetune.py`:

| Parametro | Default | Effetto |
|---|---|---|
| `--iters` | 1200 | più iterazioni vuol dire più passate sui dati, ma anche più rischio di overfitting |
| `--batch-size` | 8 | esempi per iterazione |
| `--rank` | 8 | capacità dell'adapter: più alto significa più parametri addestrabili |
| `--lr` | 1e-4 | learning rate: se la loss oscilla, abbassalo |
| `--num-layers` | 16 | quanti layer adattare (`-1` = tutti e 24 i layer del modello) |
| `--adapter-path` | `adapters/banking77` | dove salvare l'adapter |

Prima di lanciare una variante lunga, provala in piccolo: `--iters 300` sul training e
`--limit 300` sulla valutazione. Se non promette nulla su quel campione, difficilmente cambierà
sul test completo.

## Allenare con il prompt completo

Di default il training usa il prompt minimo, quindi nello scenario `ft-full` il modello riceve un
prompt diverso da quello visto in training. Per allenare con la lista delle etichette nel prompt:

```bash
uv run scripts/download.py --train-prompt full
uv run scripts/finetune.py --adapter-path adapters/banking77-fullprompt
```

Il training diventa circa 10 volte più lento, perché ogni esempio è circa 10 volte più lungo.
Rilancia `download.py` senza opzioni per tornare al prompt minimo.

## LoRA e full fine-tuning

Questo esperimento usa LoRA: i pesi del modello restano fermi e si allena un piccolo adapter
(circa lo 0,6% dei parametri). Poiché il modello è quantizzato a 4 bit, tecnicamente si tratta di QLoRA.
Il full fine-tuning, che aggiorna tutti i pesi, non si può fare su un modello a 4 bit: servirebbe la
versione non quantizzata, per esempio `Qwen/Qwen2.5-0.5B-Instruct`.
