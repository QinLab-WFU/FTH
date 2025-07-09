import argparse


def get_config():
    parser = argparse.ArgumentParser(description="DFML")

    parser.add_argument("--dataset", type=str, default="nus17", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--undataset", type=str, default="coco", help="cifar/nuswide/flickr/coco")

    parser.add_argument("--backbone", type=str, default="deit", help="see network.py")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--n_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--data_dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--save_dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n_classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=-1, help="mAP@topk")

    parser.add_argument("--num_train", type=int, default=10000, help="number of training samples")

    parser.add_argument("--alpha", type=float, default=0.8, help="number of training samples")
    parser.add_argument("--beta", type=float, default=0.6, help="number of training samples")

    parser.add_argument("--center_path", type=str, default='c', help="number of training samples")
    parser.add_argument("--epoch_change", type=int, default=9, help="number of training samples")

    parser.add_argument(
        "--model_name", type=str, default="deit_small_distilled_patch16_224.fb_in1k", help="timm pre-trained model name"
    )
    parser.add_argument("--type_of_distance", type=str, default="cosine", help="cosine/euclidean")
    parser.add_argument("--type_of_triplets", type=str, default="all", help="all/semi-hard/hard")
    parser.add_argument("--margin", type=float, default=0.25, help="margin for triplet loss")

    return parser.parse_args()
