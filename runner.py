"""TODO"""
import torch
from argparse import ArgumentParser
import models.model as model


def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument(
        "--permission-type",
        dest="permission_type",
        help="Test permission",
        default="N/A",
    )
    parser.add_argument(
        "--useful-reviews",
        dest="useful_reviews",
        help="Number of useful reviews",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--saved-data",
        dest="saved_data",
        help="Saved embeddings, train/test & vocab data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-reviews",
        dest="saved_reviews",
        help="Saved review data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-predicted-reviews",
        dest="saved_predicted_reviews",
        help="Saved predicted review data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--model-checkpoint",
        dest="model_checkpoint",
        help="Saved model file",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        help="Output directory",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--stemmer", dest="stemmer", help="Apply word stemmer", default="porter"
    )
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--hidden-size", help="Hidden vector size", type=int, default=128
    )
    parser.add_argument(
        "--init-weight",
        help="Initial value range for model parameters",
        type=float,
        default=0.08,
    )
    parser.add_argument("--output-size", help="Model output size", type=int, default=1)
    parser.add_argument(
        "--grad-clip", help="Gradient clipping value", type=int, default=5
    )
    parser.add_argument(
        "--dropout", help="Dropout for classifier", type=float, default=0
    )
    parser.add_argument("--dropoutrec", help="Dropout for RNNs", type=float, default=0)
    parser.add_argument(
        "--learning-rate-decay",
        help="Learning rate decay parameter",
        type=float,
        default=0.985,
    )
    parser.add_argument(
        "--learning-rate-decay-after",
        help="Start learning decay after given epoch number",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--print-every", help="Print control parameter", type=int, default=100
    )
    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def main():
    """TODO"""
    args = parse_arguments()
    model.run(args)


if __name__ == "__main__":
    main()
