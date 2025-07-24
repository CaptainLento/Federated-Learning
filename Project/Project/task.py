"""Project01: A Flower / PyTorch app."""

# Import librerie
from collections import OrderedDict # mantenere l'ordine delle chiavi quando si caricano i pesi.
import torch # Libreria principale per il deep learning.
import torch.nn as nn # Moduli per costruire reti neurali.
import torch.nn.functional as F # Funzioni di attivazione e altre operazioni.
from flwr_datasets import FederatedDataset # Libreria per caricare e partizionare dataset per FL.
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner # Strategie di partizionamento dei dati.
from torch.utils.data import DataLoader # Per caricare i dati in batch.
from torchvision.transforms import Compose, Normalize, ToTensor # Per trasformare le immagini.



# Definisce la classe Net, che rappresenta il modello di rete neurale.
# Eredita da nn.Module, la classe base per tutti i moduli neurali in PyTorch. (simple CNN)
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        # Viene chiamato una volta per inizializzare il modello, sia sul server che su ogni client.
        super(Net, self).__init__()
        # primo layer convoluzionale: 1 canale di input (immagini in scala di grigi), 6 canali di output, kernel 5x5 .
        self.conv1 = nn.Conv2d(1, 6, 5)
        # primo layer max pooling: dimensione finestra 2x2, Stride di 2 (riduce la dimensione spaziale).
        self.pool = nn.MaxPool2d(2, 2)
        # secondo layer convoluzionale: 6 canali di input (output del conv1), 16 canali di output, kernel 5x5 .
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Definisce il primo layer completamente connesso (fully connected):
        # - L'input è 16 * 29 * 29 neuroni (risultato del flattening delle feature map dopo i layer convoluzionali e di pooling).
        # - L'output è 120 neuroni.
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        # Definisce il secondo layer completamente connesso:
        # - Input 120 neuroni (output di fc1).
        # - Output 84 neuroni.
        self.fc2 = nn.Linear(120, 84)
        # Definisce il terzo layer completamente connesso:
        # - Input 84 neuroni (output di fc2).
        # - Output 10 neuroni. Questo sarà il numero di classi di output
        # - Nota: non so perche con 10 classi si ottengono risultati migliori nei test (dovrebbe essere 4)
        self.fc3 = nn.Linear(84, 10)

    # forward pass del modello (x è il tensore di input, l'immagine).
    # Viene chiamato ogni volta che si passa un input (un'immagine o un batch di immagini) al modello per ottenere le previsioni.
    def forward(self, x):
        # Applica il primo layer convoluzionale (conv1), funzione di attivazione ReLU, e il pooling.
        x = self.pool(F.relu(self.conv1(x)))
        # Applica il secondo layer convoluzionale (conv2), funzione di attivazione ReLU, e il pooling.
        x = self.pool(F.relu(self.conv2(x)))
        # Rimodella il tensore `x` (flattening) per prepararlo per i layer completamente connessi.
        x = x.view(-1, 16 * 29 * 29)
        # Applica il primo layer completamente connesso (fc1), poi la funzione di attivazione ReLU.
        x = F.relu(self.fc1(x))
        # Applica il secondo layer completamente connesso (fc2), poi la funzione di attivazione ReLU.
        x = F.relu(self.fc2(x))
        # Applica l'ultimo layer completamente connesso (fc3) e restituisce l'output finale del modello.
        # Questi output sono i univoci per ogni classe. (server, client)
        return self.fc3(x)



# Definisce una funzione per ottenere le trasformazioni PyTorch per le immagini.
# Viene chiamato all'inizializzazione del dataloader, sia sul server che sui client.
def get_transforms():
    # Crea una sequenza di trasformazioni:
    # - ToTensor(): Converte l'immagine PIL (Python Imaging Library) o NumPy ndarray in un tensore PyTorch.
    # - Normalize((0.5,), (0.5,)): Normalizza i valori dei pixel. Per immagini a canale singolo, (bianco e nero img)
    #   normalizza i pixel in modo che siano compresi tra -1 e 1 (media 0.5, deviazione standard 0.5).
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5,), (0.5,))]
    )

    # Funzione interna per applicare le trasformazioni a un batch di dati.
    # Viene chiamata dal metodo .with_transform() del dataset.
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    # Restituisce la funzione `apply_transforms` che verrà usata per trasformare i dati.
    return apply_transforms

fds = None  # Cache FederatedDataset
# Inizializza una variabile globale `fds`. Questa verrà usata per memorizzare l'istanza di FederatedDataset
# in modo che venga caricata una sola volta, anche se `load_data` viene chiamata più volte.



# Definisce una funzione per caricare i dati per un client specifico.
# Viene chiamata una volta per ogni client durante la sua inizializzazione (in client_app.py).
# - partition_id: L'ID della partizione (client) per cui caricare i dati.
# - num_partitions: Il numero totale di partizioni (client) nel sistema.
def load_data(partition_id: int, num_partitions: int):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Se fds è None, significa che il dataset non è ancora stato caricato.
        # Inizializza un DirichletPartitioner per creare partizioni non-IID (non identicamente distribuite).
        # - partition_by="label": La partizione avviene in base alle etichette delle classi
        # - alpha=3.0: Parametro che controlla il grado di non-IID. Valori più piccoli di alpha creano partizioni più eterogenee,
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=3.0)
        # Carica il dataset "Falah/Alzheimer_MRI" da Hugging Face e lo partiziona usando DirichletPartitioner.
        fds = FederatedDataset(
            dataset="Falah/Alzheimer_MRI",
            # Applica il partizionatore solo allo split "train".
            partitioners={"train": partitioner},
        )
    # Carica la partizione specifica per il client corrente (identificato da partition_id).
    # partition gestisce in modo autonomo la non riutilizzabilita di un dato gia presente in un altra partition
    partition = fds.load_partition(partition_id)
    # Suddivide la partizione locale del client in un set di training (80%) e un set di test/validation locale (20%).
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Applica le trasformazioni definite da `get_transforms` al dataset partizionato localmente.
    partition_train_test = partition_train_test.with_transform(get_transforms())
    # Crea un DataLoader per il set di training locale:
    # - batch_size=32: I dati verranno caricati in batch di 32 immagini.
    # - shuffle=True: I dati verranno mescolati in ogni epoca.
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    # Crea un DataLoader per il set di test/validation locale:
    # - batch_size=32: I dati verranno caricati in batch di 32 immagini.
    # - shuffle=False (implicito): I dati non verranno mescolati per la valutazione.
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    # Restituisce i due dataloader, uno per l'addestramento e uno per la valutazione locale del client.
    return trainloader, testloader



# Definisce la funzione per addestrare il modello.
# Viene chiamata dai client durante il processo `fit` (vedere client_app.py).
# - net: L'istanza del modello Net.
# - trainloader: Il DataLoader per il set di training.
# - epochs: Il numero di epoche locali per l'addestramento.
# - lr: Il learning rate per l'ottimizzatore.
# - device: Il dispositivo su cui eseguire l'addestramento (es. "cpu" o "cuda").
def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    # Sposta il modello sulla GPU se disponibile.
    net.to(device)
    # Definisce la funzione di perdita (Cross-Entropy per la classificazione) e l'ottimizzatore Adam per aggiornare i pesi del modello.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    # Inizializza la perdita cumulativa.
    running_loss = 0.0
    # Ciclo per il numero di epoche locali
    for _ in range(epochs):
        # Ciclo su ogni batch di dati nel trainloader.
        for batch in trainloader:
            images = batch["image"] # Estrae le immagini dal batch.
            labels = batch["label"] # Estrae le etichette dal batch.
            optimizer.zero_grad() # Azzera i gradienti accumulati dall'ottimizzatore dalle iterazioni precedenti.
            loss = criterion(net(images.to(device)), labels.to(device)) # Calcola l'output del modello, poi la perdita.
            loss.backward() # Calcola i gradienti della perdita rispetto a tutti i parametri addestrabili del modello.
            optimizer.step() # Aggiorna i pesi del modello usando i gradienti calcolati e il learning rate.
            running_loss += loss.item() # Aggiunge il valore della perdita del batch alla perdita cumulativa.
    # Calcola la perdita media per epoca.        
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss



# Definisce la funzione per valutare il modello.
# Viene chiamata dai client durante il processo `evaluate` (vedere client_app.py)
# e dal server per la valutazione centralizzata (vedere server_app.py).
# - net: L'istanza del modello Net.
# - testloader: Il DataLoader per il set di test/validation.
# - device: Il dispositivo su cui eseguire la valutazione.
def test(net, testloader, device):
    """Validate the model on the test set."""
    # Sposta il modello sulla GPU se disponibile.
    net.to(device)
    # Definisce la funzione di perdita.
    criterion = torch.nn.CrossEntropyLoss()
    # Inizializza contatori per le previsioni corrette e la perdita totale.
    correct, loss = 0, 0.0 
     # Imposta il modello in modalità valutazione (disabilita dropout, batchnorm, ecc.).
    net.eval()
    with torch.no_grad():
        # Ciclo su ogni batch di dati nel testloader.
        for batch in testloader:
            images = batch["image"].to(device) # Estrae le immagini e le sposta sul device.
            labels = batch["label"].to(device) # Estrae le etichette e le sposta sul device.
            outputs = net(images) # Esegue il forward pass del modello per ottenere le previsioni.
            loss += criterion(outputs, labels).item() # Calcola e accumula la perdita.
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item() # Calcola il numero di previsioni corrette.
    # Calcola l'accuratezza come (previsioni corrette / numero totale di campioni).
    accuracy = correct / len(testloader.dataset)
    # Calcola la perdita media per batch.
    loss = loss / len(testloader)
    return loss, accuracy



# Definisce una funzione per ottenere i pesi del modello.
# Viene chiamata dai client dopo l'addestramento locale (in client_app.py)
# e dal server per inizializzare i parametri globali (in server_app.py).
def get_weights(net):
    # Restituisce una lista di array NumPy, dove ogni array rappresenta i pesi di un layer del modello.
    # - net.state_dict().items(): Ottiene un iteratore su coppie (nome_layer, tensore_pesi) del modello.
    # - val.cpu().numpy(): Sposta il tensore dalla GPU alla CPU (se era su GPU) e lo converte in un array NumPy.
    return [val.cpu().numpy() for _, val in net.state_dict().items()]



# Definisce una funzione per impostare i pesi del modello.
# Viene chiamata dai client prima dell'addestramento o della valutazione (in client_app.py)
# e dal server per la valutazione centralizzata (in server_app.py).
# - net: L'instance del modello Net.
# - parameters: Una lista di array NumPy contenente i nuovi pesi.
def set_weights(net, parameters):
    # Crea un dizionario ordinato mappando i nomi dei layer ai nuovi pesi.
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # Carica i nuovi pesi nel modello.
    net.load_state_dict(state_dict, strict=True)

