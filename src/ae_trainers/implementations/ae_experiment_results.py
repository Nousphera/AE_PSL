

class ExperimentResultsAE:
    def __init__(self, validation_mode):
        self.epochs: list[int] = []
        self.train_metric: list = []
        self.test_metric: list = []
        self.final_test_metric: float = -1
        self.using_validation_set = validation_mode != 'none'

    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)

    def to_json(self):
        return {
            "final_test_loss": self.final_test_metric,
            "epochs": self.epochs,
            "train_loss": self.train_metric,
            "validation_loss" if self.using_validation_set else "test_loss" : self.test_metric,
        }