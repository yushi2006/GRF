def setup_data():
    """Downloads the mosi_data.pkl file directly from Google Drive."""
    print("--- 1. Setting up dataset ---")
    file_id = "1A9Y2Pl4pugVMPzN7RJRtyw_7-7-j5JoT"
    data_dir = "data"
    output_filename = "mosi_data.pkl"
    data_file_path = os.path.join(data_dir, output_filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(data_file_path):
        print(f"Dataset not found. Downloading '{output_filename}' from Google Drive to '{data_dir}'...")
        gdown.download(id=file_id, output=data_file_path, quiet=False)
        print("Download complete.")
    else:
        print("Dataset already exists.")

    print("--- Dataset setup complete ---\n")
    return data_file_path

def load_mosi_data(data_path: str, num_classes: int):
    """Loads data and processes labels for either 2-class or 7-class classification."""
    print(f"Loading data from {data_path} for {num_classes}-class classification...")
    if num_classes not in [2, 7]:
        raise ValueError("This pipeline only supports 2-class or 7-class classification.")

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    processed_data = {}
    for split in ["train", "valid", "test"]:
        split_data = data[split]
        raw_labels = split_data["labels"].squeeze()

        if num_classes == 2:
            final_labels = torch.tensor((raw_labels >= 0).astype(int), dtype=torch.long)
        else:
            final_labels = torch.tensor(raw_labels + 3, dtype=torch.long)

        modalities = {
            "text": torch.tensor(split_data["text"], dtype=torch.float32),
            "audio": torch.tensor(split_data["audio"], dtype=torch.float32),
            "vision": torch.tensor(split_data["vision"], dtype=torch.float32),
        }
        processed_data[split] = {
            "labels": final_labels,
            "modalities": {name: ModalityData(features=features, lengths=torch.full((features.shape[0],), features.shape[1], dtype=torch.long), name=name) for name, features in modalities.items()}
        }
        print(f"Loaded '{split}' split with {len(final_labels)} samples ({num_classes}-class).")
    return processed_data
