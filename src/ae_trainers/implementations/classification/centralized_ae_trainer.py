
from trainers.implementations.experiment_trainer import ExperimentTrainer


class CentralizedAETrainer(ExperimentTrainer):
    def _perform_epoch(self, experiment_results, auto_encoder, device, dataloader, epoch_nr, optimizer, loss_fn,
                       **kwargs):


        is_in_test_mode = optimizer is None
        auto_encoder.eval() if is_in_test_mode else auto_encoder.train()

        total_loss = 0
        nr_of_batches = len(dataloader)

        # The dataloader now yields PRE-CALCULATED activations
        for batch in dataloader:
            if not is_in_test_mode:
                optimizer.zero_grad()

            # TensorDataset yields a tuple (tensor,), so we unpack it
            # We move the activations to GPU here for training
            activations = batch[0].to(device)

            reconstructed = auto_encoder(activations)
            loss = loss_fn(reconstructed, activations)

            total_loss += loss.item()

            if not is_in_test_mode:
                loss.backward()
                optimizer.step()

        if nr_of_batches > 0:
            total_loss /= nr_of_batches

        if experiment_results is not None: experiment_results.add_results(epoch_nr, total_loss, is_in_test_mode)


        return total_loss

    def train_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            base_model=kwargs['base_model'],
            auto_encoder=kwargs['auto_encoder'],
            split_layer=kwargs['split_layer'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=kwargs['optimizer'],
            loss_fn=kwargs['loss_fn']
        )

    def test_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            base_model=kwargs['base_model'],
            auto_encoder=kwargs['auto_encoder'],
            split_layer=kwargs['split_layer'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=None,
            loss_fn=kwargs['loss_fn']
        )