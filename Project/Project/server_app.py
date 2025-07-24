"""Project01: A Flower / PyTorch app."""

from typing import List, Tuple # Per suggerimenti di tipo.
from flwr.common import Context, ndarrays_to_parameters, Metrics # Tipi e utilità comuni.
from flwr.server import ServerApp, ServerAppComponents, ServerConfig # Componenti server di Flower.
from flwr.server.strategy import FedAvg # La strategia di aggregazione Federated Averaging.
from project01.task import Net, get_weights, set_weights, test, get_transforms # modello Net e le funzioni di supporto da task.py.
from datasets import load_dataset # Per caricare il dataset globale per la valutazione centralizzata.
from torch.utils.data import DataLoader # Per caricare i dati in batch.
import json



# Funzione `get_evaluate_fn`: Restituisce una funzione di callback per la valutazione del modello globale.
# Viene chiamata una volta all'inizializzazione della strategia FedAvg.
def get_evaluate_fn(testloader, device):
    """Return a callback that evaluate the global model accuracy"""
    # Funzione interna `evaluate`: Questa è la funzione di callback effettiva.
    # Viene chiamata dal server dopo l'aggregazione di un round per valutare il modello globale.
    # - server_round: Il numero del round corrente.
    # - parameters_ndarrays: I pesi del modello globale aggregato dal server (come array NumPy).
    # - config: Configurazione aggiuntiva.
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using the test dataset not used for train"""
        # Inizializza un nuovo modello.
        net = Net()
        set_weights(net, parameters_ndarrays) # Imposta i pesi del modello con i parametri globali.
        net.to(device) # Sposta il modello sul dispositivo (CPU, in questo caso del server).
        # Esegue la valutazione sul dataset di test globale.
        loss, accuracy = test(net, testloader, device)
        # Restituisce la perdita e l'accuratezza della valutazione centralizzata.
        return loss, {"Centralized_accuracy": accuracy} 
        
    return evaluate # Restituisce la funzione di callback.



# Funzione `handle_fit_metrics`: Gestisce le metriche inviate dai client dopo la fase `fit`.
# Viene chiamata dal server dopo ogni round di addestramento quando i client restituiscono le loro metriche.
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # `metrics` è una lista di tuple, dove ogni tupla contiene (numero_di_esempi, metriche_del_client).
    for _, m in metrics:
        print(m) # Stampa le metriche di ogni client.

    return {} # Restituisce un dizionario vuoto o aggregato se necessario.



# Funzione `on_fit_config`: Regola la configurazione da inviare ai client per il prossimo round di addestramento.
# Viene chiamata dal server all'inizio di ogni round di addestramento, prima che i client ricevano i parametri.
def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.01 # Learning rate iniziale.
    # Regola dinamicamente il learning rate in base al numero del round.
    if server_round > 5:
        lr = 0.005
    if server_round > 15:
        lr = 0.003 
    if server_round > 25:
        lr = 0.001      
    return {"lr": lr} # Restituisce il learning rate da inviare ai client.



# Funzione `weighted_average`: Aggrega le metriche di valutazione ricevute dai client.
# Viene chiamata dal server dopo ogni round di valutazione (se `fraction_evaluate` è > 0).
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # `metrics` è una lista di tuple (num_esempi, metriche_del_client).
    # Calcola l'accuratezza media pesata.
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Restituisce l'accuratezza media pesata.
    return {"accuracy": sum(accuracies) / total_examples}



# Funzione `server_fn`: Questa è la funzione principale che Flower chiama per creare l'istanza del server.
# Viene chiamata una volta all'avvio del server Flower.
def server_fn(context: Context):
    # Legge le configurazioni dal contesto di esecuzione (derivate da pyproject.toml).
    num_rounds = context.run_config["num-server-rounds"] # Numero totale di round.
    fraction_fit = context.run_config["fraction-fit"] # Frazione di client da selezionare per l'addestramento.

    # Inizializza i parametri del modello per il server.
    ndarrays = get_weights(Net()) # Ottiene i pesi di un modello Net appena inizializzato.
    parameters = ndarrays_to_parameters(ndarrays) # Converte i pesi nel formato richiesto da Flower.
    
    # Carica il dataset di test globale per la valutazione centralizzata del modello.
    # Questo dataset non è usato per l'addestramento federato, ma per monitorare le prestazioni del modello globale.
    testset = load_dataset("Falah/Alzheimer_MRI")["test"] 
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32) # Crea un dataloader.

    # Definisce la strategia di apprendimento federato. Qui viene usata FedAvg (Federated Averaging).
    strategy = FedAvg(
        fraction_fit=fraction_fit, # Frazione di client che partecipano all'addestramento.
        fraction_evaluate=1.0, # Frazione di client che partecipano alla valutazione (se attivata).
        min_available_clients=2, # Numero minimo di client richiesti per avviare un round.
        initial_parameters=parameters, # I parametri iniziali del modello globale.
        # Funzioni di aggregazione delle metriche:
        evaluate_metrics_aggregation_fn=weighted_average, # Aggrega le metriche di valutazione dei client.
        fit_metrics_aggregation_fn=handle_fit_metrics, # Gestisce le metriche di addestramento dei client.
        on_fit_config_fn=on_fit_config, # Funzione per inviare configurazioni ai client prima del fit.
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"), # Funzione per la valutazione centralizzata.
    )
    # Definisce la configurazione del server (principalmente il numero di round).
    config = ServerConfig(num_rounds=num_rounds)

    # Restituisce i componenti dell'applicazione server (strategia e configurazione).
    return ServerAppComponents(strategy=strategy, config=config)

# Crea l'istanza dell'applicazione ServerApp.
# Flower utilizzerà questa 'app' per coordinare l'intera federazione.
app = ServerApp(server_fn=server_fn)
