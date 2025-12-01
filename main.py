import os
import argparse
import json
from collections import Counter, defaultdict
import string
import nltk
import tqdm
import numpy as np
import scipy.sparse as sp
import time
import bottleneck as bn
import numba as nb
from typing import List
import multiprocessing
from functools import reduce

from bm25.bm25_index import (
        Bm25Index, 
        index_chunk,
        compute_token_scores, 
        precompute_token_scores

)

def main(args):
    collection_path = args.collection_path
    if args.verbose:
        assert collection_path

    if args.index:
        Bm25Index.build_index(args.collection_path, args.output_path, args.index_name)
        return

    start = time.time()
    index = Bm25Index(
        collection_path=collection_path,
        index_root=args.output_path,
        index_name=args.index_name,
        precompute_scores=args.precompute_scores,
    )
    end = time.time()
    print(f"Loaded index in {(end - start):.2f} seconds")

    queries = []
    with open(os.path.join(os.environ["DATA_PATH"], args.queries), "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries.append(query)

    print("Warming up...")
    for query in tqdm.tqdm(queries[:10]):
        index.search(query, k=10, verbose=False)

    print("Benchmarking...")
    query_batches = np.array_split(queries, len(queries) // args.batch_size)
    latencies = []
    if args.n is not None:
        query_batches = query_batches[: args.n]
    for query_batch in tqdm.tqdm(query_batches):
        start = time.time()
        if args.batch_size == 1:
            index.search(
                query_batch[0],
                k=args.k,
                weight_by_frequency=args.weight_by_frequency,
                verbose=args.verbose,
            )
        else:
            index.search_all(
                query_batch,
                k=args.k,
                weight_by_frequency=args.weight_by_frequency,
                verbose=args.verbose,
            )
        end = time.time()
        latencies.append(end - start)
    print(f"Average latency: {(np.mean(latencies) * 1e3):.2f} ms")
    print(f"Median latency: {(np.median(latencies) * 1e3):.2f} ms")
    print(f"P95 latency: {(np.percentile(latencies, 95) * 1e3):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 indexer")
    parser.add_argument("--index", action="store_true", default=False)
    parser.add_argument(
        "--collection_path", type=str, default=None, help="Path to collection"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--index_name", type=str, required=True, help="Index name")
    parser.add_argument("-k", type=int, default=1000, help="Number of docs to return")
    parser.add_argument("-n", type=int, default=None, help="Number of queries")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--precompute_scores",
        action="store_true",
        default=False,
        help="Precomputes scores (faster but takes more memory)",
    )
    parser.add_argument(
        "--weight_by_frequency",
        action="store_true",
        default=False,
        help="Weight query terms by frequency",
    )
    parser.add_argument(
        "--queries", type=str, required=True, help="Path to queries tsv file"
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose")
    args = parser.parse_args()
    main(args)
