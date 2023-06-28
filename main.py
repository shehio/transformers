from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset("cnn_dailymail", version="3.0.0")
    print(dataset)
    print(f"Features: {dataset['train'].column_names}")
