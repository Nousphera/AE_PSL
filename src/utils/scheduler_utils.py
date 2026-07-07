import torch

from models.vision_transformer.implementations.unimodal.image_classification.models import ClientModel


def get_optimizer_and_scheduler(model, global_args):
    optimizer = torch.optim.Adam(model.parameters(), lr=global_args['start_lr'])

    return optimizer, get_scheduler(optimizer, global_args)


def get_optimzer_and_scheduler_for_seperate_ae_finetuning(model: ClientModel, global_args):
    if global_args['compression_method'] == 'ae':
        if global_args['concurrent_mse_alignment']:
            # In this case we must split the optimizers, one for the main model and one for the AE module. We want to make sure that the AE module is not updated by the main optimizer, and vice versa.

            # 1. Get the IDs of the AE module parameters
            ae_params_ids = set(id(p) for p in model.ae_module.parameters())

            # 2. Filter the rest of the model
            # This catches conv_proj, class_token, pos_embedding, and blocks
            main_params = [p for p in model.parameters() if id(p) not in ae_params_ids]

            optimizer_main = torch.optim.Adam(main_params, lr=global_args['start_lr'])
            optimizer_ae = torch.optim.Adam(model.ae_module.parameters(), lr=global_args['ae_finetune_lr'])
            scheduler_main = get_scheduler(optimizer_main, global_args)
            scheduler_ae = torch.optim.lr_scheduler.ConstantLR(optimizer_ae, factor=1.0, total_iters=0)
        else:
            # Using the full models parameters here
            optimizer_main = torch.optim.Adam(model.parameters(), lr=global_args['start_lr'])
            scheduler_main = get_scheduler(optimizer_main, global_args)

            # AE optimizer and scheduler are not being used, but we need to create dummy ones to return
            # However they are used for warmup if the ae_type is not identity, so we need to create them in that case
            if global_args['ae_type'] != 'identity':
                optimizer_ae = torch.optim.Adam(model.ae_module.parameters(), lr=global_args['ae_finetune_lr'])
                scheduler_main = get_scheduler(optimizer_main, global_args)
                scheduler_ae = torch.optim.lr_scheduler.ConstantLR(optimizer_ae, factor=1.0, total_iters=0)
            else:
                optimizer_ae = None
                scheduler_ae = None
    else:
        # If we are not using an AE, then we can just return the main optimizer and scheduler, and dummy ones for the AE
        optimizer_main = torch.optim.Adam(model.parameters(), lr=global_args['start_lr'])
        scheduler_main = get_scheduler(optimizer_main, global_args)
        optimizer_ae = None
        scheduler_ae = None

    return optimizer_main, scheduler_main, optimizer_ae, scheduler_ae


def get_scheduler(optimizer, global_args):
    if global_args['scheduler'] == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    if global_args['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=global_args['scheduler_step_size'])
    elif global_args['scheduler'] == 'cosine':
        # print("Warning, using CosineAnnealingLR default eta min lr 5e-5")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_args['nr_of_epochs'],
                                                               eta_min=5e-5)
    else:
        raise NotImplementedError(f"Unsupported scheduler: {global_args['ae_pretrain_scheduler']}")
    return scheduler


def get_ae_pretrain_optimizer_and_scheduler(model, global_args):

    if global_args['ae_pretrain_optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=global_args['ae_pretrain_start_lr'])
    else:
        raise NotImplementedError(f"Unsupported optimizer: {global_args['ae_pretrain_optimizer']}")

    if global_args['ae_pretrain_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=global_args['ae_pretrain_scheduler_step_size'])
    elif global_args['ae_pretrain_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_args['ae_pretrain_epochs'], eta_min=global_args['ae_pretrain_cosine_eta_min'])
    else:
        raise NotImplementedError(f"Unsupported scheduler: {global_args['ae_pretrain_scheduler']}")
    return optimizer, scheduler