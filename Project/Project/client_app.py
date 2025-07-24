"""Project01: A Flower / PyTorch app."""

import torch # Libreria PyTorch.
from flwr.client import ClientApp, NumPyClient # Componenti client del framework Flower.
from flwr.common import Context # Oggetto per accedere al contesto di esecuzione di Flower.
from project01.task import Net, get_weights, load_data, set_weights, test, train # modello Net e le funzioni di supporto da task.py.
import json # Usato per serializzare metriche complesse in stringhe JSON.



# Definisce la classe FlowerClient che eredita da NumPyClient,
class FlowerClient(NumPyClient):
    # Il costruttore viene chiamato una volta quando l'istanza del client viene creata (in client_fn).
    # Inizializza il modello, i dataloader e le impostazioni specifiche del client.
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        # Determina il dispositivo (GPU o CPU) su cui eseguire il modello.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Sposta il modello sul dispositivo selezionato.
        self.net.to(self.device)

    # Metodo `fit`: Viene chiamato dal server Flower all'inizio di ogni round di addestramento (se selezionato).
    # Riceve i parametri globali del modello dal server e una configurazione (es. learning rate).
    def fit(self, parameters, config):
        # Imposta i pesi del modello locale con i parametri ricevuti dal server.
        set_weights(self.net, parameters)
        # Esegue l'addestramento locale del modello sul proprio dataset.
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config['lr'], # Ottiene il learning rate dalla configurazione inviata dal server.
            self.device,
        )
        # Prepara le metriche da inviare al server (in questo caso, la perdita di addestramento).
        complex_metrics = {"train_loss": train_loss} # mettere accuracy dei client
        complex_metrics_str = json.dumps(complex_metrics) # Serializza le metriche in una stringa JSON.
        # Restituisce i pesi aggiornati del modello locale, la dimensione del dataset di addestramento locale
        # e le metriche di addestramento al server.
        return (
            get_weights(self.net), # I pesi del modello dopo l'addestramento locale.
            len(self.trainloader.dataset), # Il numero di campioni nel dataset di addestramento locale.
            {"client_metrics": complex_metrics_str}, # Metriche aggiuntive (loss).
        )

    # Metodo `evaluate`: Viene chiamato dal server Flower, facoltativamente, dopo ogni round di addestramento
    # o in un round di valutazione dedicato, per valutare le prestazioni del modello locale.
    def evaluate(self, parameters, config):
        # Imposta i pesi del modello locale con i parametri globali ricevuti dal server.
        set_weights(self.net, parameters)
        # Esegue la valutazione del modello sul proprio set di validazione locale.
        loss, accuracy = test(self.net, self.valloader, self.device)
        # Restituisce la perdita di valutazione, la dimensione del dataset di validazione locale
        # e l'accuratezza al server.
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}



# Funzione `client_fn`: Questa è la funzione principale che Flower chiama per creare un'istanza di client.
# Viene chiamata una volta per ogni client simulato all'inizio della simulazione.
def client_fn(context: Context):
    # Inizializza un nuovo modello.
    net = Net()
    # Ottiene l'ID della partizione e il numero totale di partizioni dalla configurazione del nodo.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # Carica i dati per questo client, ottenendo i dataloader per addestramento e validazione.
    trainloader, valloader = load_data(partition_id, num_partitions)
    # Ottiene il numero di epoche locali dalla configurazione di esecuzione.
    local_epochs = context.run_config["local-epochs"]

    # Restituisce un'istanza di FlowerClient. ( .to_client() )
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()



# Flower ClientApp: Il punto di ingresso per l'applicazione client.
# Flower utilizzerà questa 'app' per gestire i client nella federazione.
app = ClientApp(
    client_fn,
)
