from deepflow.data import DiscreteDirectionalDataset


def test_data(dataset: DiscreteDirectionalDataset):
    assert len(dataset) == 8
    image, groundtruth, mask = dataset[0]
    assert image.shape == (1, 2, 2)
    assert groundtruth.shape == (2, 2, 8)
    assert mask.shape == (2, 2)
