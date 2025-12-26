import os
import datasets

from .AbsTask import AbsTask


class BeIRTask(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        try:
            from beir import util
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        try:
            if self.description["beir_name"].startswith("cqadupstack"):
                raise ImportError("CQADupstack is incompatible with latest BEIR")
            from beir.datasets.data_loader_hf import HFDataLoader as BeirDataLoader
        except ImportError:
            from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader

        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        dataset = self.description["beir_name"]
        dataset, sub_dataset = dataset.split("/") if "cqadupstack" in dataset else (dataset, None)

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            download_path = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
            data_path = util.download_and_unzip(url, download_path)
            data_path = f"{data_path}/{sub_dataset}" if sub_dataset else data_path
            corpus, queries, relevant_docs = BeirDataLoader(
                data_folder=data_path
            ).load(split=split)
            
            # Convert Dataset objects to dictionaries if needed for compatibility with BEIR
            from datasets import Dataset
            if isinstance(corpus, Dataset):
                corpus = {item['id']: {k: v for k, v in item.items() if k != 'id'} for item in corpus}
            if isinstance(queries, Dataset):
                queries = {item['id']: item['text'] for item in queries}
            
            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = relevant_docs
        self.data_loaded = True
