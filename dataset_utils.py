from dataset import FaceAgeDataset
from torch.utils.data import Dataset, DataLoader


def create_datasets(
    dataset_type="train",
    data_dir="/home/jessekim",
    transform=None,
    mask_info=None,
    bs=64,
    shuffle=True,
    num_workers=4,
    data_source="wiki",
):
    """returns train_loader if train_data is True, else test_loader. Returns validation_loader as well either way"""

    path = f"/home/jessekim/data/{data_source}_{dataset_type}.csv"
    print(path)
    # train or test loader
    df = FaceAgeDataset(
        csv_path=path,
        data_dir=data_dir,
        transform=transform,
        mask_info=mask_info,
        data_source=data_source,
    )

    loader = DataLoader(df, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    return loader
