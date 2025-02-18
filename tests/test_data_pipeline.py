import pytest
import shutil
import yaml
from src.data_pipeline import ensure_annotations_present, split_dataset, create_yaml_yolo, validate_dataset


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Create a temporary dataset structure for testing"""
    dataset_path = tmp_path / "test_dataset"
    dataset_path.mkdir()

    # Create some test images and annotations
    for i in range(10):
        # Create test image
        img_path = dataset_path / f"image_{i}.jpg"
        img_path.touch()

        # Create test annotation
        ann_path = dataset_path / f"image_{i}.txt"
        with open(ann_path, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

    yield dataset_path
    # Cleanup
    shutil.rmtree(dataset_path)


def test_ensure_annotations_present(temp_dataset_path):
    """Test annotation presence check"""
    # Should return True as annotations were created in fixture
    assert ensure_annotations_present(temp_dataset_path)

    # Remove all annotations and test again
    for ann_file in temp_dataset_path.glob("*.txt"):
        ann_file.unlink()

    assert not ensure_annotations_present(temp_dataset_path)


def test_split_dataset(temp_dataset_path):
    """Test dataset splitting functionality"""
    split_dataset(temp_dataset_path, 0.6, 0.2)

    # Check if split directories were created
    assert (temp_dataset_path / "train" / "images").exists()
    assert (temp_dataset_path / "train" / "labels").exists()
    assert (temp_dataset_path / "valid" / "images").exists()
    assert (temp_dataset_path / "valid" / "labels").exists()
    assert (temp_dataset_path / "test" / "images").exists()
    assert (temp_dataset_path / "test" / "labels").exists()

    # Check split ratios (approximately)
    train_images = list((temp_dataset_path / "train" / "images").glob("*.jpg"))
    valid_images = list((temp_dataset_path / "valid" / "images").glob("*.jpg"))
    test_images = list((temp_dataset_path / "test" / "images").glob("*.jpg"))

    assert len(train_images) == 6  # 60% of 10
    assert len(valid_images) == 2  # 20% of 10
    assert len(test_images) == 2  # 20% of 10


def test_create_yaml_yolo(temp_dataset_path):
    """Test YAML creation for YOLO training"""
    # Create mock classes
    mock_classes = ["class1", "class2"]

    create_yaml_yolo(temp_dataset_path, mock_classes)

    yaml_path = temp_dataset_path / "yolo.yaml"
    assert yaml_path.exists()

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    assert config["nc"] == 2
    assert config["names"] == {0: "class1", 1: "class2"}
    assert config["train"] == "./train/images"
    assert config["val"] == "./valid/images"
    assert config["test"] == "./test/images"


def test_validate_dataset(temp_dataset_path):
    """Test dataset validation"""
    # First split the dataset
    split_dataset(temp_dataset_path, 0.6, 0.2)

    # Should pass validation without raising exception
    validate_dataset(temp_dataset_path)

    # Create an invalid label file with corresponding image
    invalid_label_path = temp_dataset_path / "train" / "labels" / "invalid.txt"
    invalid_image_path = temp_dataset_path / "train" / "images" / "invalid.jpg"
    invalid_image_path.touch()
    with open(invalid_label_path, "w") as f:
        f.write("invalid format")  # Wrong format

    # Should raise ValueError for invalid format
    with pytest.raises(ValueError):
        validate_dataset(temp_dataset_path)
