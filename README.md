# 🧠 Federated Learning in Medicine: A paradigm for advancing Healthcare and maintaining data privacy  

## 🌟 Introduzione 
Questo repository contiene il codice sorgente del progetto di tesi incentrato sull'implementazione di un sistema di Apprendimento Federato (Federated Learning - FL) per l'analisi di immagini di Risonanza Magnetica (MRI) cerebrale, con l'obiettivo di supportare la diagnosi di condizioni legate all'Alzheimer.  
Il progetto sfrutta il framework Flower 🌸 per orchestrare il processo di addestramento distribuito e PyTorch 🔥 per la costruzione e l'addestramento del modello di deep learning.  
L'apprendimento federato consente di addestrare un modello globale su dati distribuiti su più client (es. ospedali, centri di ricerca) senza che i dati sensibili lascino mai la loro posizione originale, garantendo così privacy 🔒 e conformità normativa.  

## ✨ Caratteristiche Principali
* **Apprendimento Federato:** Implementazione basata su Flower per l'addestramento di modelli su dati decentralizzati.
* **Modello CNN:** Utilizzo di una Rete Neurale Convoluzionale (CNN) sviluppata in PyTorch per la classificazione delle immagini MRI.
* **Dataset Medicale:** Addestramento e valutazione su un dataset di immagini MRI del cervello (`Falah/Alzheimer_MRI`).
* **Configurazione Flessibile:** Parametri di simulazione configurabili (numero di round, frazione di client, epoche locali, learning rate dinamico).

## 📂 Struttura del Progetto

Il progetto è organizzato come segue:
```bash
├── client_app.py           # 👨‍⚕️ Implementazione del client: gestisce l'addestramento e la valutazione locale.
├── server_app.py           # 💻 Implementazione del server: orchestra i round di comunicazione e aggrega i modelli.
├── task.py                 # 🧠 Definisce l'architettura della CNN (`Net`), le funzioni di training/testing e caricamento dati.
├── pyproject.toml          # ⚙️ File di configurazione del progetto e delle dipendenze.
├── README.md               # 📄 Questo file.
└── venv/                   # 🌳 Ambiente virtuale (se creato localmente).  
```
## 📊 Dataset

Il dataset utilizzato è `Falah/Alzheimer_MRI`. Questo dataset contiene immagini di risonanza magnetica cerebrale suddivise in diverse categorie (e.g., soggetti sani, diverse fasi dell'Alzheimer).  
**Nota Importante:** La natura distribuita dell'apprendimento federato, con la suddivisione del dataset globale in partizioni per ciascun client, può portare a una quantità relativamente limitata di dati per l'addestramento locale. Ciò nonostante, il sistema è stato progettato per dimostrare l'efficacia del FL anche in contesti di dati frammentati.

## 🛠️ Prerequisiti

Assicurati di avere Python 3.10 o superiore installato sul tuo sistema.

## 🚀 Installazione

1.  **Clona il repository:**
    ```bash
    git clone [https://github.com/CaptainLento/Federated-Learning.git](https://github.com/CaptainLento/Federated-Learning.git)
    cd Federated-Learning
    ```

2.  **Crea e attiva un ambiente virtuale (raccomandato):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Linux/macOS
    venv\Scripts\activate     # Su Windows
    ```

3.  **Installa le dipendenze del progetto:**
    Le dipendenze sono specificate nel file `pyproject.toml`.
    ```bash
    pip install -e .
    ```

## ▶️ Utilizzo

Per avviare la simulazione di apprendimento federato, esegui il seguente comando dalla radice del progetto:
```bash
flwr run .
```

Il comando avvierà il server Flower e crea i client in un ambiente di simulazione locale.  
La configurazione dei round di training, della frazione di client e delle epoche locali è definita in pyproject.toml (es. 30 round, 3 epoche locali, campionamento del 50% dei client).  
Durante l'esecuzione, potrai osservare nei log:
* La perdita (loss) e l'accuratezza del modello globale.  
* Le metriche di training (train_loss) inviate dai singoli client.  
* L'andamento della Centralized_accuracy e della distributed_accuracy nel corso dei round.  

### 🔺 Risultati Chiave della Simulazione:
La simulazione è stata configurata per 30 round di comunicazione e addestramento, con una strategia di learning rate dinamica.  
* Accuratezza Iniziale: Il modello non addestrato ha mostrato un'accuratezza centralizzata del 0.0%.  
* Accuratezza Finale: Al termine dei 30 round, il modello globale ha raggiunto un'accuratezza centralizzata di circa 88.28% sul set di test del server.  
* L'accuratezza aggregata dalle valutazioni distribuite sui client è stata di circa 89.67%.  

### 🔺 Convergenza della Loss e durata del processo: 
La perdita centralizzata è scesa da 2.337 a circa 1.523, mentre la perdita distribuita ha mostrato un andamento simile, convergendo.  
Durata Totale: L'intera simulazione si è completata in circa 404.54 secondi.  
Questi risultati dimostrano la capacità del sistema di apprendere efficacemente da dati distribuiti, migliorando notevolmente le prestazioni del modello iniziale.  

### 🔺 Considerazioni Aggiuntive:  
È importante ribadire che questo algoritmo di apprendimento federato è concepito come un ausilio decisionale per i professionisti medici, non come un sostituto della diagnosi umana. La responsabilità finale e l'interpretazione clinica rimangono sempre di competenza del medico.  
Inoltre, si prevede che l'utilizzo di un set di dati maggiore e più diversificato possa portare a un ulteriore miglioramento delle performance del modello, sia in termini di accuratezza che di capacità di generalizzazione, mitigando l'impatto della frammentazione dei dati in un ambiente federato.  

### 🔺 Licenza
Questo progetto è distribuito sotto licenza Apache-2.0. Per maggiori dettagli, consulta il file di licenza presente nel repository.

### 🔺 Riconoscimenti
Ringraziamenti a:  
* Al Relatore e al Correlatore per la preziosa guida e il costante supporto durante questo percorso di tesi
* Il team di Flower per l'eccellente framework di apprendimento federato.
* Il team di PyTorch per la libreria di deep learning.
* I creatori del dataset Falah/Alzheimer_MRI per aver reso disponibili i dati.
