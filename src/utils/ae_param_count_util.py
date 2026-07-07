import json


from models.auto_encoder import AE_REGISTRY



def calculate_params():
    ae_types = [
        "single_non_linear", "double_non_linear", "double_non_linear_straight",
        "double_non_linear_hourglass", "triple_non_linear", "conv_spatial_AE"
    ]
    latent_dims = [192, 96, 48, 32, 24, 16]
    input_dim = 768  # Standard for vit_b_32

    results = {}

    for ae_type in ae_types:
        results[ae_type] = {}
        for l_dim in latent_dims:
            # Prepare the global_args dict as expected by your classes
            global_args = {
                'model': 'vit_b_32',
                'ae_type': ae_type,
                'ae_latent_dim': l_dim
            }

            try:
                # Instantiate model (no checkpoint needed for param counting)
                if ae_type == 'identity':
                    model = AE_REGISTRY[ae_type]()
                else:
                    model = AE_REGISTRY[ae_type](
                        input_dim=input_dim,
                        latent_dim=l_dim,
                        global_args=global_args
                    )

                # Count parameters
                count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                results[ae_type][l_dim] = count

            except Exception as e:
                results[ae_type][l_dim] = f"Error: {str(e)}"

    # Save to JSON
    with open('../../data/ae_parameter_counts.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Parameter counts generated and saved to ae_parameter_counts.json")


if __name__ == "__main__":
    calculate_params()

    import matplotlib.pyplot as plt
    import json

    # Load the data you generated
    with open('ae_parameter_counts.json', 'r') as f:
        param_data = json.load(f)



    plt.figure(figsize=(8, 6))

    for ae_type, dims in param_data.items():
        x_params = []
        y_acc = []

        # Sort by dimension to keep lines logical
        sorted_dims = sorted(dims.keys(), key=int, reverse=True)

        for dim in sorted_dims:
            x_params.append(dims[dim])
            y_acc.append(accuracies[ae_type][dim])  # Match with your accuracy data

        plt.plot(x_params, y_acc, marker='o', label=ae_type)

    plt.xscale('log')  # Params often span orders of magnitude
    plt.xlabel('Parameter Count (Log Scale)')
    plt.ylabel('Finetune Accuracy')
    plt.title('Architecture Efficiency: Accuracy vs. Model Size')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()