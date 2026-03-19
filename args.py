
import argparse


## Parameter setting
def parameter_parser():
    parser = argparse.ArgumentParser(description="HoRM experiment arguments")

    parser.add_argument("--data_path", type=str, default="./datasets/", help="Path to multi-view datasets.")
    parser.add_argument("--dataset_id", type=int, default=None, help="Dataset ID for classification scripts. If omitted, the script uses its built-in default.")
    parser.add_argument("--dataset_clustering_id", type=int, default=None, help="Dataset ID for clustering scripts. If omitted, clustering scripts use their built-in default.")

    parser.add_argument("--input_type", type=str, default="feature", help="feature or similarity")
    # input_type； choose features or similarity graphs to learn a multi-variate heterogeneous representation.

    parser.add_argument("--fusion_type", type=str, default="weight", help="Fusion Methods: trust or average or weight or attention")
    
    parser.add_argument("--active", type=str, default="l1", help="l21 or l1")
    # the type of regularizer with Prox_h()

    parser.add_argument("--device", default="cpu", type=str, required=False)
    parser.add_argument("--fix_seed", action=argparse.BooleanOptionalAction, default=True, help="Fix random seed for reproducibility. Use --no-fix-seed to disable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clustering_seed", type=int, default=50, help="Default seed for clustering scripts.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeated runs for each dataset.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Consistency loss weight.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1, help='block') # for the example dataset, block can set 2 and more than 2
    parser.add_argument('--thre', type=float, default=0.1)
    parser.add_argument('--thre_n', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1, help='delta')
    parser.add_argument('--alpha', type=float, default=1, help='alpha')
    parser.add_argument('--beta', type=float, default=1, help='beta')

    return parser.parse_args()