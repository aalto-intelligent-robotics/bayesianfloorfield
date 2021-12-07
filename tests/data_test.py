from deepflow.data import DiscreteDirectionalDataset


def test_data(dataset: DiscreteDirectionalDataset):
    assert len(dataset) == 2
    image, groundtruth = dataset[1]
    assert image.shape == (1, 2, 2)
    assert groundtruth.shape == (8,)
