import jax
import jax.numpy as jnp

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        return f.read()

def get_vocab(dataset):
    "get the unique characters in the dataset"
    return sorted(list(set(dataset)))

def encode(s, string_to_int):
    "convert a string to a list of integers"
    return [string_to_int[ch] for ch in s]

def decode(ints, int_to_string):
    "convert a list of integers to a string"
    return ''.join(int_to_string[i] for i in ints)

def get_train_val_split(data, train_ratio=0.9):
    "split the data into train and validation sets"
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def main():
    dataset_path = 'data/tiny_shakespeare.txt'
    dataset = load_dataset(dataset_path)

    vocab = get_vocab(dataset)

    string_to_int = {ch:i for i, ch in enumerate(vocab)}
    int_to_string = {i:ch for i, ch in enumerate(vocab)}

    # print(encode(dataset[:200], string_to_int))
    # print(decode(encode(dataset[:200], string_to_int), int_to_string))

    train_data, val_data = get_train_val_split(dataset)

    # Convert encoded dataset to JAX array
    encoded_train_data = encode(train_data, string_to_int)
    encoded_val_data = encode(val_data, string_to_int)

    train_data = jnp.array(encoded_train_data)
    val_data = jnp.array(encoded_val_data)

    print(f"Train data shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"Val data shape: {val_data.shape}, dtype: {val_data.dtype}")

if __name__ == '__main__':
    main()