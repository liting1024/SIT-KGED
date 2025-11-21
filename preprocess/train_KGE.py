import os
import fire
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset, PathDataset
from pykeen.models import RGCN


def run(
        DATASET: str = "WN18RR",
        MODEL: str = "ConvE",
        DIM: int = 512,
):
    SAVE_PATH = f"preprocess/{DATASET}/{MODEL}-D{DIM}"

    dataset = get_dataset(dataset=DATASET)
    print(dataset, dataset.training.num_triples, dataset.testing.num_triples, dataset.validation.num_triples, sep="\t")

    if MODEL == 'RGCN':
        result = pipeline(
            random_seed=0,
            dataset=dataset,
            model=MODEL,
            model_kwargs={
                'embedding_dim': DIM,
            },
            training_kwargs={
                'num_epochs': 2000,
                'batch_size': 512
            },
            training_loop="slcwa",
            stopper="early",
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002,
                metric="mrr",
            ),
        )
    elif MODEL == 'RotatE':
        result = pipeline(
            random_seed=0,
            dataset=dataset,
            model=MODEL,
            model_kwargs={
                'embedding_dim': DIM,
            },
            training_kwargs={
                'num_epochs': 2000,
                'batch_size': 512
            },
            training_loop="slcwa",
            stopper="early",
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002,
                metric="mrr",
            ),
        )
    elif MODEL == 'TransE':
        result = pipeline(
            random_seed=0,
            dataset=dataset,
            model=MODEL,
            model_kwargs={
                'embedding_dim': DIM,
                'scoring_fct_norm': 1
            },
            training_kwargs={
                'num_epochs': 2000,
                'batch_size': 512
            },
            training_loop="slcwa",
            stopper="early",
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002,
                metric="mrr",
            ),
        )
    elif MODEL == 'DistMult':
        result = pipeline(
            random_seed=0,
            dataset=dataset,
            model=MODEL,
            model_kwargs={
                'embedding_dim': DIM,
            },
            training_kwargs={
                'num_epochs': 2000,
                'batch_size': 512
            },
            training_loop="slcwa",
            stopper="early",
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002,
                metric="mrr",
            ),
        )
    elif MODEL == 'ConvE':
        result = pipeline(
            random_seed=0,
            dataset=dataset,
            model=MODEL,
            model_kwargs={
                'embedding_dim': DIM,
                'input_dropout': 0.2,
                'feature_map_dropout': 0.2,
                'output_dropout': 0.3,
            },
            training_kwargs={
                'num_epochs': 2000,
                'batch_size': 512
            },
            training_loop="slcwa",
            stopper="early",
            stopper_kwargs=dict(
                patience=10,
                relative_delta=0.002,
                metric="mrr",
            ),
        )

    result.save_to_directory(SAVE_PATH)


if __name__ == "__main__":
    fire.Fire(run)
