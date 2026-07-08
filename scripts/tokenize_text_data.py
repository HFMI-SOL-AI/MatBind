import os
from argparse import ArgumentParser
from multiprocessing import Semaphore, cpu_count, Process, Queue
import numpy
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast


def _tokenize_subprocess(tokenizer: BertTokenizerFast, semaphore: Semaphore,
                         documents_queue: Queue, writer_queue: Queue):
    """
    Tokenize descriptions in a subprocess.

    :param tokenizer: A BERT tokenizer.
    :param semaphore: A semaphore to control the throttle of source documents.
    :param documents_queue: A queue from which documents are fetched.
    :param writer_queue: A queue to which tokenized documents are written.
    :return:
    """
    while True:
        item = documents_queue.get()
        semaphore.release()
        if item is None:
            break

        key, description = item
        tokens = tokenizer.tokenize(description, add_special_tokens=True)
        token_ids = numpy.array(tokenizer.convert_tokens_to_ids(tokens))

        writer_queue.put((key, token_ids))


def _write_csv_subprocess(tokenized_csv_path: str, writer_queue: Queue, dtype: numpy.dtype):
    """
    Write the tokenized data to a CSV file.

    :param tokenized_csv_path: Destination path for tokenized CSV.
    :param writer_queue: Queue containing tokenized data.
    :param dtype: Data type for the tokenized output.
    """
    tokenized_data = []

    while True:
        # Main process sends None to indicate EOF.
        item = writer_queue.get()
        if item is None:
            break

        key, token_ids = item
        tokenized_data.append({
            'id': key,
            'token_ids': ','.join(map(str, token_ids.astype(dtype)))
        })

    # Write the tokenized data to CSV
    tokenized_df = pd.DataFrame(tokenized_data)
    tokenized_df.to_csv(tokenized_csv_path, index=False)


def tokenize_csv(
        json_path: str,
        tokenized_csv_path: str,
        bert_tokenizer: str,
        cased: bool,
        processes: int = cpu_count(),
        dtype: str = 'uint16'
):
    """
    Tokenize all descriptions in a CSV dataset.

    :param json_path: Source json file with 'id' and 'description' columns.
    :param tokenized_csv_path: Destination CSV file for tokenized descriptions.
    :param bert_tokenizer: Folder that contains a BERT tokenizer.
    :param cased: Whether the tokenizer is cased.
    :param processes: Number of processes to use.
    :param dtype: What dtype to use for storing the tokenized data.
    :return:
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        bert_tokenizer, do_lower_case=not cased)

    dtype = numpy.dtype(dtype)

    assert tokenizer.vocab_size < numpy.iinfo(dtype).max, \
        f"Vocabulary size is greater than the maximum of dtype {dtype}"

    # Read the json file
    df = pd.read_json(json_path)
    # Take only required colums
    df = df[['material_id', 'structure_description']]

    semaphore = Semaphore(4096)
    tokenized_queue = Queue()

    def _description_generator():
        for _, row in df.iterrows():
            # If queue insertion is too fast, we get throttled.
            semaphore.acquire()

            yield row['material_id'], row['structure_description']

    # Create database writer.
    csv_writer = Process(
        target=_write_csv_subprocess,
        args=(tokenized_csv_path, tokenized_queue, dtype))
    csv_writer.start()

    # Create workers.
    document_queues = [Queue() for _ in range(processes)]
    workers = [Process(
        target=_tokenize_subprocess,
        args=(tokenizer, semaphore, document_queues[i], tokenized_queue)) for i in range(processes)]
    [i.start() for i in workers]

    # Distribute tasks.
    for i, item in enumerate(tqdm(_description_generator(), desc='Tokenizing descriptions')):
        document_queues[i % len(document_queues)].put(item)

    print('Notifying workers EOF...')
    for queue in document_queues:
        queue.put(None)

    # Wait for workers to finish
    for i, worker in enumerate(workers):
        print(f'Waiting for worker {i} to finish...')
        worker.join()

    print('Notifying CSV writer EOF...')
    tokenized_queue.put(None)

    # Wait for CSV write to finish
    print('Waiting for CSV writer to finish...')
    csv_writer.join()


def _main():
    parser = ArgumentParser(description='Tokenize descriptions in the json file.')

    parser.add_argument('--json_path', '-input', type=str, required=True,
                        help='Source json file with other info and "structure_description" columns.')
    parser.add_argument('--tokenized_csv_path', '-output', type=str, required=True,
                        help='Destination CSV file for tokenized descriptions.')
    parser.add_argument('--tokenizer_path', '-tokenizer', type=str, required=True,
                        help='Folder that contains a BERT tokenizer.')
    parser.add_argument('--cased', '-cased', action='store_true',
                        help='Tokenizer should be case-sensitive.')
    parser.add_argument('--processes', '-p', type=int, default=cpu_count(),
                        help='Number of processes to use.')
    parser.add_argument('--dtype', '-dtype', type=str, default='uint16',
                        help='Dtype of the stored numpy arrays.')

    args = parser.parse_args()

    assert not os.path.exists(args.tokenized_csv_path), f"Output file {args.tokenized_csv_path} already exists!"

    tokenize_csv(
        csv_path=args.json_path,
        tokenized_csv_path=args.tokenized_csv_path,
        bert_tokenizer=args.tokenizer_path,
        processes=args.processes,
        cased=args.cased,
        dtype=args.dtype,
    )


# python matbert/training/script_tokenize_perovskite.py --json_path=data.json  \
#       --tokenized_csv_path=perovskite_structure_description_tokenized_new.csv \
#       --tokenizer_path=matbert-base-cased/


# matber-base-cased should be a path within which the following files are downloaded:
# curl -# -o matbert-base-cased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json
# curl -# -o matbert-base-cased/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/vocab.txt
# curl -# -o matbert-base-cased/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/pytorch_model.bin

if __name__ == '__main__':
    _main()
