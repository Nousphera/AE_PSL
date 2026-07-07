from typing import List, Dict


class ExperimentResults:
    def __init__(self, validation_mode):
        self.epochs: List[int] = []
        self.train_metric = []
        self.test_metric = []
        self.final_test_loss: float = -1
        self.final_test_accuracy: float = -1
        self.incoming_client_communication_overhead_in_mb: Dict[str, float] = {}
        self.outgoing_client_communication_overhead_in_mb: Dict[str, float] = {}
        self.using_validation_set = validation_mode != 'none'

        # New structure to hold MSE: { "client_id": { "epoch_nr": [mse_batch_1, mse_batch_2, ...] } }
        self.client_batch_mse: Dict[str, Dict[str, List[float]]] = {}
        self.server_batch_mse: Dict[str, Dict[str, List[float]]] = {}

        self.distributed_eval = {}
        self.final_distributed_eval = {}


    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)



    def add_client_batch_mse(self, epoch_nr: int, client_id: int, batch_mse_list: List[float]):
        """
        Stores the list of batch MSEs for a specific client and epoch.
        Keys are converted to strings to ensure safe JSON serialization.
        """
        str_client_id = str(client_id)
        str_epoch = str(epoch_nr)

        if str_client_id not in self.client_batch_mse:
            self.client_batch_mse[str_client_id] = {}

        self.client_batch_mse[str_client_id][str_epoch] = batch_mse_list

    def add_server_batch_mse(self, epoch_nr: int, client_id: int, batch_mse_list: List[float]):
        """
        Stores the list of batch MSEs for a specific client and epoch.
        Keys are converted to strings to ensure safe JSON serialization.
        """
        str_client_id = str(client_id)
        str_epoch = str(epoch_nr)

        if str_client_id not in self.server_batch_mse:
            self.server_batch_mse[str_client_id] = {}

        self.server_batch_mse[str_client_id][str_epoch] = batch_mse_list

    def set_client_communication_overhead(self, epoch_nr, incoming_in_mb, outgoing_in_mb):

        self.incoming_client_communication_overhead_in_mb[str(epoch_nr)] = float(incoming_in_mb)
        self.outgoing_client_communication_overhead_in_mb[str(epoch_nr)] = float(outgoing_in_mb)

    def to_json(self):
        return {
            "final_test_loss": self.final_test_loss,
            "final_test_accuracy": self.final_test_accuracy,
            "epochs": self.epochs,
            "train_metric": self.train_metric,
            "validation_metric" if self.using_validation_set else "test_metric": self.test_metric,
            "incoming_client_communication_overhead_in_mb": self.incoming_client_communication_overhead_in_mb,
            "outgoing_client_communication_overhead_in_mb": self.outgoing_client_communication_overhead_in_mb,
            "distributed_eval_per_epoch": self.distributed_eval,
            "distributed_eval_final": self.final_distributed_eval,
            "client_batch_mse": self.client_batch_mse,  # <-- Added here
            "server_batch_mse": self.server_batch_mse,  # <-- Added here
        }

    def from_json(self, json_dict):
        self.final_test_loss = json_dict["final_test_loss"]
        self.final_test_accuracy = json_dict["final_test_accuracy"]
        self.epochs = json_dict["epochs"]
        self.train_metric = json_dict["train_metric"]
        if self.using_validation_set:
            self.test_metric = json_dict.get("validation_metric", [])
        else:
            self.test_metric = json_dict.get("test_metric", [])
        self.incoming_client_communication_overhead_in_mb = json_dict["incoming_client_communication_overhead_in_mb"]
        self.outgoing_client_communication_overhead_in_mb = json_dict["outgoing_client_communication_overhead_in_mb"]
        self.client_batch_mse = json_dict.get("client_batch_mse", {})  # <-- Added here