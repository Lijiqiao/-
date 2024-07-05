import argparse
from to_dataset import TopologyOptimizationDataset

def save_to_txt(data_file, output_file):
    dataset = TopologyOptimizationDataset(data_file)
    with open(output_file, "w") as f:
        for idx in range(len(dataset)):
            nodes, edges, loc = dataset.get_example(idx)
            loc_str = " ".join(f"{x} {y}" for x, y in loc)
            edges_str = " ".join(f"{u} {v}" for u, v in edges)
            f.write(f"{loc_str} output {edges_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TXT labels for the dataset.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the dataset file (PKL).")
    args = parser.parse_args()

    data_file = args.data_file
    output_file = data_file.replace(".pkl", ".txt")

    save_to_txt(data_file, output_file)