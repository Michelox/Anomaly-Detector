# Anomaly Detector
### Dalle vibrazioni industriali al rilevamento delle anomalie con un LSTM Autoencoder

Questo progetto esplora una domanda: **è possibile riconoscere condizioni di funzionamento anomale di un impianto industriale imparando a ricostruire le sue vibrazioni?**

Il percorso parte dai segnali di un banco di prova, ne riduce il rumore e ne estrae la componente stagionale. Una rete LSTM Autoencoder apprende a ricostruire sequenze temporali; l'errore di ricostruzione diventa poi il criterio per individuare le anomalie.

Il README accompagna la lettura della tesi **“Uso di modelli di Long Short-Term Memory per l’Anomaly Detection negli impianti industriali”**, di **Michele Ciaramidaro**, Corso di Laurea in Ingegneria Gestionale, Università degli Studi di Messina, A.A. **2023/2024**. Relatore: **Prof. Dario Bruneo**.

**[Leggi la tesi completa](Tesi%20%281%29.pdf)** · [Walkthrough](#walkthrough-della-tesi) · [Risultati](#6-risultati-e-conclusioni) · [Esecuzione](#eseguire-il-progetto)

## Walkthrough della tesi

### 1. Il problema: riconoscere un cambiamento nel funzionamento
*Capitolo 1; §2.1–2.2*

Nel monitoraggio industriale, una variazione del segnale può indicare un cambiamento nelle condizioni operative. Riconoscerla aiuta a orientare le verifiche sull'impianto, anche se un'anomalia non coincide necessariamente con un guasto.

Lo studio considera una **replica in scala di un impianto per il trasporto e l'assemblaggio di componenti automobilistiche**, composta da due motori e sei nastri che movimentano un carrello. Il banco permette di introdurre diverse perturbazioni meccaniche. Nell'esperimento descritto, le condizioni confrontate sono **con carico e senza carico**.

Tra i sensori del banco, l'analisi utilizza le misure di vibrazione attribuite nella tesi al **VTV-122**. L'obiettivo è distinguere i comportamenti delle due serie attraverso le loro caratteristiche temporali.

### 2. Dai dati grezzi alle sequenze utili
*Capitolo 2*

#### Selezione del segnale e costruzione dei dataset

Le registrazioni contengono tre colonne, corrispondenti agli assi `x`, `y` e `z`, campionate a **400 Hz**.

| Dato sperimentale | Valore riportato nella tesi |
| --- | ---: |
| Osservazioni nella registrazione normale | 9.632.614 |
| Osservazioni nella registrazione anomala | 531.370 |
| Osservazioni normali dopo l'esclusione dei picchi | 3.100.000 |
| Asse selezionato | z |

L'asse **z** viene mantenuto perché risulta il più sollecitato dalle vibrazioni. I picchi nella serie normale, attribuiti a errori di configurazione dei sensori, vengono esclusi prima della costruzione dei dataset.

Il §2.3 descrive questo scambio di osservazioni:

- **Training set:** la serie normale ripulita, con le ultime 1.000 osservazioni sostituite dalle prime 1.000 della serie anomala.
- **Test set:** la serie anomala, con le prime 1.000 osservazioni sostituite dalle ultime 1.000 della serie normale.

Il training descritto nella tesi contiene quindi anche una piccola porzione della registrazione anomala. Le Figure 1 e 2 mostrano rispettivamente i segnali iniziali e i dataset costruiti.

#### Riduzione del rumore con il filtro di Kalman

Il filtro di Kalman aggiorna ricorsivamente una stima del segnale combinando previsione e misura. Serve a ridurre il rumore di acquisizione prima dell'analisi delle sequenze.

Il codice del §2.4.3 impiega un modello scalare con `A = H = 1`, covarianza del rumore di processo `Q = 0.01` e covarianza del rumore di misura `R = 0.1`. La Figura 3 mostra i segnali filtrati.

#### Estrazione della stagionalità

La decomposizione separa la serie in **trend, stagionalità e residui**. Poiché i trend delle due condizioni risultano simili, lo studio aggiunge al dataset la componente stagionale, cercando nei pattern ricorrenti informazioni utili a distinguere i comportamenti.

L'ingresso del modello comprende così **due feature: il segnale z filtrato e la stagionalità**, coerentemente con la forma dell'input riportata nel Capitolo 6. La Figura 4 illustra la decomposizione.

#### Normalizzazione

Le feature vengono normalizzate con `MinMaxScaler` prima dell'ingresso nella rete. Nel pacchetto Python attuale lo scaler viene adattato esclusivamente al training set e poi applicato al test set.

### 3. Perché combinare LSTM e autoencoder?
*Capitoli 3, 4 e 5*

Le vibrazioni sono **serie temporali**: l'ordine delle misure contiene informazione. Dopo un'introduzione all'apprendimento automatico, la tesi passa dalle reti ricorrenti alle **Long Short-Term Memory**, che usano uno stato di memoria e meccanismi di selezione dell'informazione per modellare dipendenze nel tempo.

L'**autoencoder** introduce invece un compito di ricostruzione:

- L'**encoder** comprime la sequenza in una rappresentazione interna.
- Il **decoder** usa quella rappresentazione per ricostruire la sequenza di partenza.
- La differenza tra ingresso e ricostruzione misura quanto bene il modello rappresenta quel comportamento.

Combinando i due approcci, la rete impara una rappresentazione delle sequenze di vibrazione. L'ipotesi del detector è che sequenze diverse dai pattern appresi producano errori di ricostruzione maggiori.

### 4. Costruire e addestrare il modello
*§6.2–6.5*

Il segnale viene suddiviso in **finestre di 100 campioni**, equivalenti a **0,25 secondi a 400 Hz**. Il codice della tesi avanza di un campione alla volta, producendo finestre sovrapposte. Ogni ingresso ha forma `(100, 2)`: 100 passi temporali e due feature.

| Passaggio | Strato | Forma dell'output per finestra |
| --- | --- | --- |
| Ingresso | Sequenza di vibrazione e stagionalità | `100 × 2` |
| Encoder | LSTM, 16 unità, restituisce la sequenza | `100 × 16` |
| Rappresentazione compressa | LSTM, 8 unità | `8` |
| Espansione temporale | RepeatVector, 100 ripetizioni | `100 × 8` |
| Decoder | LSTM, 8 unità, restituisce la sequenza | `100 × 8` |
| Decoder | LSTM, 16 unità, restituisce la sequenza | `100 × 16` |
| Ricostruzione | TimeDistributed(Dense(2)) | `100 × 2` |

Le LSTM utilizzano l'attivazione `tanh`. L'output ha la stessa forma dell'input, così da confrontare direttamente ogni sequenza con la sua ricostruzione.

| Parametro | Esperimento descritto nella tesi |
| --- | ---: |
| Lunghezza della finestra | 100 campioni |
| Epoche | 20 |
| Batch size | 50 |
| Quota di validazione | 25% |
| Loss di ricostruzione | Mean Absolute Error (MAE) |

Il pacchetto attuale compila il modello con **Adam** e loss **MAE**. La Figura 11 della tesi mostra l'andamento della loss di training e validazione, utile per osservare l'apprendimento e l'eventuale divergenza tra le due curve.

### 5. Trasformare l'errore in una decisione
*§6.1, §6.6–6.7*

Dopo l'addestramento, il modello ricostruisce le sequenze di training e di test. Il detector misura l'errore assoluto medio tra i valori osservati e quelli ricostruiti.

Per una finestra con `T` passi temporali e `F` feature, il punteggio utilizzato nel pacchetto attuale è:

```math
\mathrm{MAE}(X,\hat X)=\frac{1}{TF}\sum_{t=1}^{T}\sum_{f=1}^{F}|X_{t,f}-\hat X_{t,f}|
```

La tesi sceglie come **soglia il massimo errore di ricostruzione sul training set**. Una sequenza viene segnalata come anomala quando il suo errore supera questa soglia.

La **Figura 12** mostra la distribuzione della loss sul training; la **Figura 13** confronta le distribuzioni di training e test con la soglia. È il passaggio che collega l'apprendimento della rete alla decisione finale.

### 6. Risultati e conclusioni
*Capitolo 7, Figura 14*

La tesi riporta i seguenti risultati sul test set dell'esperimento:

| Metrica | Risultato riportato |
| --- | ---: |
| Precision | 1,00 |
| Recall | 1,00 |
| F1-score | 1,00 |

Il testo delle conclusioni riporta inoltre un'accuratezza del **100%**. La valutazione considera anomala la serie di test ad eccezione delle prime 1.000 osservazioni, provenienti dalla serie normale.

Nel protocollo descritto, questi risultati indicano una separazione completa delle classi valutate. Il contributo dello studio è mostrare come **pulizia del segnale, stagionalità e ricostruzione con LSTM** possano lavorare insieme per distinguere condizioni operative differenti.

I risultati si riferiscono alle condizioni sperimentali e al test set descritti nella tesi. Per valutarne la generalizzazione servono nuove acquisizioni e un protocollo di test indipendente.

## Orientarsi nel repository

| Risorsa | Contenuto |
| --- | --- |
| [Tesi completa](Tesi%20%281%29.pdf) | Contesto scientifico, metodo, figure e risultati |
| [Preprocessing](src/anomaly_detector/preprocessing.py) | Lettura CSV, split, filtro di Kalman, stagionalità e finestre |
| [Modello](src/anomaly_detector/model.py) | Architettura LSTM Autoencoder |
| [Scoring](src/anomaly_detector/scoring.py) | MAE, soglia e classificazione delle finestre |
| [CLI](src/anomaly_detector/cli.py) | Addestramento, scoring e salvataggio degli output |
| [Test](tests/test_preprocessing.py) | Controlli su dati sintetici |
| [Configurazione](pyproject.toml) | Dipendenze e installazione |
| [Script dell'esperimento](legacy/Anomaly%20detector.py) | Implementazione conservata per consultazione |

## Eseguire il progetto

### Installazione

Servono **Python 3.10+** e un ambiente compatibile con TensorFlow.

```bash
git clone https://github.com/Michelox/Anomaly-Detector.git
cd Anomaly-Detector
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"
```

Su Windows, attivare l'ambiente con `.venv\Scripts\activate`.

### Preparazione degli input

I CSV sperimentali non sono inclusi nel repository: occorre fornire i propri dati, già ripuliti dagli eventuali errori di acquisizione.

Fornire due CSV: uno con misure normali e uno con la registrazione anomala. Il programma legge la **terza colonna**, corrispondente all'asse z, che deve contenere valori numerici finiti. L'intestazione è prevista per impostazione predefinita; aggiungere `--no-header` se entrambi i file ne sono privi.

Collocare i file, ad esempio, in `data/normal.csv` e `data/anomalous.csv`. Con il periodo predefinito, ciascuna serie risultante dallo split deve contenere almeno **7.200 campioni**, necessari alla decomposizione stagionale. Il programma riserva inoltre 1.000 campioni normali al test. Il periodo è espresso in campioni: 3.600 campioni corrispondono a 9 secondi a 400 Hz. La pipeline lavora offline, perché la decomposizione stagionale usa l'intera serie fornita.

### Addestramento e rilevamento

```bash
python -m anomaly_detector \
  --normal-csv data/normal.csv \
  --anomalous-csv data/anomalous.csv \
  --output-dir outputs/run1
```

I valori predefiniti includono finestre di 100 campioni, passo 100, 20 epoche e batch size 50. Per usare finestre sovrapposte come nel codice della tesi, aggiungere `--window-stride 1`: su milioni di osservazioni questa scelta aumenta sensibilmente il consumo di memoria e il tempo di calcolo.

Per consultare tutte le opzioni e avviare i test:

```bash
python -m anomaly_detector --help
python -m pytest -q
```

### Output

| File | Contenuto |
| --- | --- |
| `model.keras` | Modello addestrato |
| `scaler.joblib` | Scaler adattato ai dati di training |
| `train_scores.csv` | Posizione delle finestre e loss sul training |
| `test_scores.csv` | Loss, soglia e segnalazione delle finestre anomale |
| `loss.png` | Grafico degli errori di ricostruzione e della soglia |
| `run.json` | Parametri, soglia e conteggi dell'esecuzione |

In `test_scores.csv`, `is_anomaly` rappresenta la decisione del modello. `overlaps_anomalous_source` indica soltanto se la finestra interseca la registrazione anomala: è un'informazione sulla provenienza, non un'etichetta di guasto verificata.

---

**Riferimento:** Michele Ciaramidaro, *Uso di modelli di Long Short-Term Memory per l’Anomaly Detection negli impianti industriali*, Università degli Studi di Messina, A.A. 2023/2024. Per lo sviluppo teorico e i riferimenti bibliografici, consultare la [tesi completa](Tesi%20%281%29.pdf).
