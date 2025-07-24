"""Project01: A Flower / PyTorch app."""

# Import libreries
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# Definisce la classe Net, che rappresenta il modello di rete neurale.
# Eredita da nn.Module, la classe base per tutti i moduli neurali in PyTorch. (simple CNN)
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        # primo layer convoluzionale: 1 canale di input, 6 canali di output, kernel 5x5 .
        self.conv1 = nn.Conv2d(1, 6, 5)
        # primo layer max pooling: dimensione finestra 2x2, Stride di 2
        self.pool = nn.MaxPool2d(2, 2)
        # primo layer convoluzionale: 6 canale di input, 16 canali di output, kernel 5x5 .
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
        # - Output 10 neuroni
        self.fc3 = nn.Linear(84, 10)

    # forward pass del modello (x è il tensore di input (l'immagine)..
    def forward(self, x):
        # Applica il primo layer convoluzionale (conv1), funzione di attivazione ReLU, e il pooling.
        x = self.pool(F.relu(self.conv1(x)))
        # Applica il secondo layer convoluzionale (conv1), funzione di attivazione ReLU, e il pooling.
        x = self.pool(F.relu(self.conv2(x)))
        # Rimodella il tensore `x` (flattening) per prepararlo per i layer completamente connessi.
        # 16 * 29 * 29 è la dimensione delle feature map appiattite.
        x = x.view(-1, 16 * 29 * 29)
        # Applica il primo layer completamente connesso (fc1), poi la funzione di attivazione ReLU.
        x = F.relu(self.fc1(x))
        # Applica il secondo layer completamente connesso (fc2), poi la funzione di attivazione ReLU.
        x = F.relu(self.fc2(x))
        # Applica l'ultimo layer completamente connesso (fc3) e restituisce l'output finale del modello.
        return self.fc3(x)

# Definisce una funzione per ottenere le trasformazioni PyTorch per le immagini.
# Crea una sequenza di trasformazioni:
    # - ToTensor(): Converte l'immagine in un tensore PyTorch.
    # - Normalize((0.5,), (0.5,)): Normalizza i valori dei pixel. Per immagini a canale singolo,
    # - normalizza i pixel in modo che siano compresi tra -1 e 1 (mean=0.5, std=0.5).
def get_transforms():
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5,), (0.5,))]
    )

    
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # Funzione interna per applicare le trasformazioni a un batch di dati.
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    # Restituisce il batch con le immagini trasformate.
    return apply_transforms


fds = None  # Cache FederatedDataset
# Inizializza una variabile globale `fds` a None. Questa verrà usata per memorizzare l'istanza di FederatedDataset
# in modo che venga caricata una sola volta, anche se `load_data` viene chiamata più volte.


# Definisce una funzione per caricare i dati per un client specifico.
    # - partition_id: L'ID della partizione (client) per cui caricare i dati.
    # - num_partitions: Il numero totale di partizioni (client) nel sistema.
def load_data(partition_id: int, num_partitions: int):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=3.0)
        # Inizializza un DirichletPartitioner.
        # - num_partitions: Il numero totale di client.
        # - partition_by="label": La partizione avviene in base alle etichette delle classi.
        # - alpha=3.0: parametro che controlla il grado di non-IID. Valori più piccoli di alpha
        #   creano partizioni più eterogenee (non-IID).
        fds = FederatedDataset(
            dataset="Falah/Alzheimer_MRI",
            # nome del dataset da caricare da Hugging Face.
            partitioners={"train": partitioner},
            # Applica il partizionatore Dirichlet solo allo split "train" del dataset.
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Suddivide la partizione del client in un set di training (80%) e un set di test/validation locale (20%).
    # Ridefinisce le trasformazioni PyTorch (potrebbe essere riutilizzata la funzione `get_transforms` per evitare duplicazioni).
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5,), (0.5,))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        # Ridefinisce la funzione interna per applicare le trasformazioni
        return batch
    # Applica le trasformazioni al dataset partizionato localmente.
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    # Crea un DataLoader per il set di training locale:
    # - batch_size=32: I dati verranno caricati in batch di 32 immagini.
    # - shuffle=True: I dati verranno mescolati in ogni epoca.
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    # Crea un DataLoader per il set di test/validation locale:
    # - batch_size=32: I dati verranno caricati in batch di 32 immagini.
    # - shuffle=False (implicito): I dati non verranno mescolati per la valutazione.
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    # dataloader per test e train
    return trainloader, testloader


# Definisce la funzione per addestrare il modello.
    # - net: L'istanza del modello Net.
    # - trainloader: Il DataLoader per il set di training.
    # - epochs: Il numero di epoche locali per l'addestramento.
    # - lr: Il learning rate per l'ottimizzatore.
    # - device: Il dispositivo su cui eseguire l'addestramento (es. "cpu" o "cuda").
def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Definisce l'ottimizzatore Adam, che aggiornerà i pesi del modello.
    # - net.parameters(): Tutti i parametri addestrabili del modello.
    # - lr: Il learning rate.
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            # Ciclo su ogni batch di dati nel trainloader.
            images = batch["image"]
            # Estrae le immagini dal batch.
            labels = batch["label"]
            # Estrae le etichette dal batch.
            optimizer.zero_grad()
            # Azzera i gradienti accumulati dall'ottimizzatore dalle iterazioni precedenti. Essenziale ad ogni batch.
            loss = criterion(net(images.to(device)), labels.to(device))
            # Calcola l'output del modello per le immagini (spostate sul device),
            # poi calcola la perdita confrontando l'output con le etichette (spostate sul device).
            loss.backward()
            # Calcola i gradienti della perdita rispetto a tutti i parametri addestrabili del modello.
            optimizer.step()
            # Aggiorna i pesi del modello usando i gradienti calcolati e il learning rate.
            running_loss += loss.item()
            # Aggiunge il valore della perdita del batch alla perdita cumulativa.

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


# Definisce la funzione per valutare il modello.
    # - net: L'istanza del modello Net.
    # - testloader: Il DataLoader per il set di test/validation.
    # - device: Il dispositivo su cui eseguire la valutazione.
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            # Ciclo su ogni batch di dati nel testloader.
            images = batch["image"].to(device)
            # Estrae le immagini e le sposta sul device.
            labels = batch["label"].to(device)
            # Estrae le etichette e le sposta sul device.
            outputs = net(images)
            # Esegue il forward pass del modello.
            loss += criterion(outputs, labels).item()
            # Calcola e accumula la perdita.
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # Calcola il numero di previsioni corrette:
            # - torch.max(outputs.data, 1)[1]: Trova l'indice della classe con la probabilità più alta.
            # - == labels: Confronta le previsioni con le etichette reali.
            # - .sum().item(): Somma il numero di corrispondenze corrette.
    accuracy = correct / len(testloader.dataset)
    # Calcola l'accuratezza come (previsioni corrette / numero totale di campioni nel dataset di test).
    loss = loss / len(testloader)
    # Calcola la perdita media per batch.
    return loss, accuracy
    # Restituisce la perdita media e l'accuratezza.
    # Definisce una funzione per ottenere i pesi del modello.


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
     # Restituisce una lista di array NumPy, dove ogni array rappresenta i pesi di un layer del modello.
     # - net.state_dict().items(): Ottiene un iteratore su coppie (nome_layer, tensore_pesi) del modello.
     # - val.cpu().numpy(): Sposta il tensore dalla GPU alla CPU (se era su GPU) e lo converte in un array NumPy.


# Definisce una funzione per impostare i pesi del modello.
    # - net: L'istanza del modello Net.
    # - parameters: Una lista di array NumPy contenente i nuovi pesi.
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    # Carica i nuovi pesi nel modello.
    # - strict=True: Richiede che tutte le chiavi nel state_dict fornito corrispondano esattamente alle chiavi del modello.
