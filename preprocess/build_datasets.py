import os
import json

from pykeen.datasets import get_dataset, PathDataset
from pykeen.sampling import BasicNegativeSampler, BernoulliNegativeSampler

import torch
import torch.nn as nn

from node_label import HoCN, get_high_order_adj
from torch_sparse import SparseTensor

import pandas as pd
import numpy as np

np.random.seed(0)
from sentence_transformers import SentenceTransformer
import fire


def run(
    DATASET="fb15k237",  # WN18RR; codexsmall, codexmedium; fb15k237
    MODEL="TransE-D512",
    SENTENCE_PATH="/home/HF_Model/SentenceTransformer/all-MiniLM-L6-v2",
):
    SAVE_PATH = f"preprocess/{DATASET}"  # KGE save path
    DATA_PATH = f"data/{DATASET}"
    dataset = get_dataset(dataset=DATASET)
    print(dataset.training)

    # dataset.to_directory_binary()
    # print(dataset._base_path)

    trained_model = torch.load(
        f"{SAVE_PATH}/{MODEL}/trained_model.pkl", weights_only=False
    )
    print(trained_model)

    template = {
        # TODO change template for different datasets, but now same
        "WN18RR": {
            "KGED": {
                "instruction": "You are an expert in knowledge graph reasoning. Your task is to classify a given knowledge graph triple into one of the following categories: Triple Correct, Head Entity Error, Tail Entity Error, Relation Error.",
                "input": "Is this true: ({}, {}, {})?",
            },
        },
        "codexsmall": {
            "KGED": {
                "instruction": "You are an expert in knowledge graph reasoning. Your task is to classify a given knowledge graph triple into one of the following categories: Triple Correct, Head Entity Error, Tail Entity Error, Relation Error.",
                "input": "Is this true: ({}, {}, {})?",
            },
        },
        "codexmedium": {
            "KGED": {
                "instruction": "You are an expert in knowledge graph reasoning. Your task is to classify a given knowledge graph triple into one of the following categories: Triple Correct, Head Entity Error, Tail Entity Error, Relation Error.",
                "input": "Is this true: ({}, {}, {})?",
            },
        },
        "fb15k237": {
            "KGED": {
                "instruction": "You are an expert in knowledge graph reasoning. Your task is to classify a given knowledge graph triple into one of the following categories: Triple Correct, Head Entity Error, Tail Entity Error, Relation Error.",
                "input": "Is this true: ({}, {}, {})?",
            },
        },
    }
    # template = json.loads(json.dumps(template), object_hook=lambda x: type('Object', (), x)())

    """
    supplement text and detail for entites and relations
    """
    if DATASET in ["WN18RR", "WN18"]:
        entity2text_PAHT = f"preprocess/raw/{DATASET}/entity2text.txt"
        df = pd.read_csv(
            entity2text_PAHT, sep="\t", header=None, names=["entity", "text"]
        )

        df["id"] = df["entity"].map(dataset.entity_to_id)
        df = (
            df.dropna()
        )  # For WN18RR, pykeen/dataset.num_entities 40559 < raw/entity2text.txt 40943; id is not consecutive
        df["id"] = df["id"].astype(int)

        split_result = df["text"].str.split(",", n=1, expand=True)
        df["text"] = split_result[0].str.strip()
        df["detail"] = split_result[1].str.strip()

        entity_id_to_text = df.set_index("id")["text"].to_dict()
        entity_id_to_detail = df.set_index("id")["detail"].to_dict()

    elif DATASET in ["UMLS"]:
        entity_df = pd.read_csv(
            f"preprocess/raw/{DATASET}/entity2text.txt",
            sep="\t",
            header=None,
            names=["entity", "text"],
        )
        entity_df["id"] = entity_df["entity"].map(dataset.entity_to_id)
        entity_id_to_text = entity_df.set_index("id")["text"].to_dict()

        entity_df = pd.read_csv(
            f"preprocess/raw/{DATASET}/entity2textlong.txt",
            sep="\t",
            header=None,
            names=["entity", "text"],
        )
        entity_df["id"] = entity_df["entity"].map(dataset.entity_to_id)
        entity_id_to_detail = entity_df.set_index("id")["text"].to_dict()

        relation_df = pd.read_csv(
            f"preprocess/raw/{DATASET}/relation2text.txt",
            sep="\t",
            header=None,
            names=["relation", "text"],
        )
        relation_df["id"] = relation_df["relation"].map(dataset.relation_to_id)
        relation_df = relation_df.dropna(subset=["id"])
        relation_df["id"] = relation_df["id"].astype(int)
        relation_id_to_text = relation_df.set_index("id")["text"].to_dict()

    elif DATASET in [
        "codexsmall",
        "codexmedium",
    ]:
        entity_df = pd.read_json(f"preprocess/raw/{DATASET}/entities.json")
        entity_df = entity_df.reset_index()
        entity_df["id"] = entity_df["index"].map(dataset.entity_to_id)
        entity_df = entity_df.dropna(subset=["id"])
        entity_df["id"] = entity_df["id"].astype(int)

        entity_id_to_text = entity_df.set_index("id")["label"].to_dict()
        entity_id_to_detail = entity_df.set_index("id")["description"].to_dict()

        relation_df = pd.read_json(
            f"preprocess/raw/{DATASET}/relations.json", orient="index"
        )
        relation_df = relation_df.reset_index()
        relation_df["id"] = relation_df["index"].map(dataset.relation_to_id)
        relation_df = relation_df.dropna(subset=["id"])
        relation_df["id"] = relation_df["id"].astype(int)

        relation_id_to_text = relation_df.set_index("id")["label"].to_dict()

    elif DATASET in ["FB15k-237", "fb15k237"]:
        entity_df = pd.read_csv(
            f"preprocess/raw/{DATASET}/entity.csv",
            sep=",",
            header=0,
            names=["entity", "text", "desc"],
        )
        entity_df["id"] = entity_df["entity"].map(dataset.entity_to_id)
        entity_df = entity_df.dropna(subset=["id"])
        entity_df["id"] = entity_df["id"].astype(int)
        entity_id_to_text = entity_df.set_index("id")["text"].to_dict()
        entity_id_to_detail = entity_df.set_index("id")["desc"].to_dict()

        relation_df = pd.read_csv(
            f"preprocess/raw/{DATASET}/MKGL_relation.txt",
            sep="\t",
            header=None,
            names=["relation", "text"],
        )
        relation_df["id"] = relation_df["relation"].map(dataset.relation_to_id)
        relation_df = relation_df.dropna(subset=["id"])
        relation_df["id"] = relation_df["id"].astype(int)
        relation_id_to_text = relation_df.set_index("id")["text"].to_dict()

    def get_SBert_embed():
        """
        Get Embedding from SentenceTransformer
        Save entity_embeddings.pth
        """
        print(os.listdir(SENTENCE_PATH))  # check model files

        sentence_model = SentenceTransformer(
            SENTENCE_PATH,
            # cache_folder='./all-MiniLM-L6-v2', use_auth_token=''
        )
        print(sentence_model)
        embedding_dim = 384
        entity_embeddings = torch.zeros(dataset.num_entities, embedding_dim)
        relation_embeddings = torch.zeros(dataset.num_relations, embedding_dim)

        for id, detail in entity_id_to_detail.items():
            entity_embedding = sentence_model.encode(detail)
            entity_embedding = torch.tensor(entity_embedding, dtype=torch.float32)
            entity_embeddings[id] = entity_embedding
        for id, detail in relation_id_to_text.items():
            relation_embedding = sentence_model.encode(detail)
            relation_embedding = torch.tensor(relation_embedding, dtype=torch.float32)
            relation_embeddings[id] = relation_embedding

        return entity_embeddings, relation_embeddings

    def get_KGE_embed():
        """
        Embeddings from TransE
        """
        entity_embeddings = (
            trained_model.entity_representations[0](indices=None).detach().cpu().numpy()
        )
        relation_embeddings = (
            trained_model.relation_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )
        entity_embeddings = torch.tensor(entity_embeddings)
        relation_embeddings = torch.tensor(relation_embeddings)

        print(entity_embeddings.shape)
        print(relation_embeddings.shape)

        return entity_embeddings, relation_embeddings

    def generate_triple_sample():
        """
        BernoulliNegativeSampler    1:2
        1 for entity-corrupt, 1 for relation-corrupt
        """

        def generate_sample(data, SAMPLE_TYPE="N"):
            triple_df = pd.DataFrame(
                data.mapped_triples, columns=["head_id", "relation_id", "tail_id"]
            )

            if SAMPLE_TYPE == "N":
                # ======= entity neg ========
                ent_negative_sampler = BernoulliNegativeSampler(
                    mapped_triples=data.mapped_triples,
                    num_negs_per_pos=1,
                    filtered=True,
                )
                ent_negative_sample, filter_mask = ent_negative_sampler.sample(
                    data.mapped_triples
                )
                ent_negative_sample = ent_negative_sample.view(-1, 3)

                ent_neg_df = pd.DataFrame(
                    ent_negative_sample, columns=["head_id", "relation_id", "tail_id"]
                )
                ent_neg_df["neg_type"] = [
                    "Head Entity Error" if h != h_neg else "Tail Entity Error"
                    for (h, r, t), (h_neg, r_neg, t_neg) in zip(
                        data.mapped_triples.tolist(), ent_negative_sample.tolist()
                    )
                ]
                # ======= relation neg ========
                rel_negative_sampler = BasicNegativeSampler(
                    corruption_scheme=["relation"],
                    mapped_triples=data.mapped_triples,
                    num_negs_per_pos=1,
                    filtered=True,
                )
                rel_negative_sample, filter_mask = rel_negative_sampler.sample(
                    data.mapped_triples
                )
                rel_negative_sample = rel_negative_sample.view(-1, 3)
                rel_neg_df = pd.DataFrame(
                    rel_negative_sample, columns=["head_id", "relation_id", "tail_id"]
                )
                rel_neg_df["neg_type"] = "Relation Error"
                triple_df = pd.concat([ent_neg_df, rel_neg_df], ignore_index=True)

            triple_df["head"] = triple_df["head_id"].map(entity_id_to_text)
            triple_df["tail"] = triple_df["tail_id"].map(entity_id_to_text)
            if DATASET in ["codexsmall", "fb15k237", "UMLS", "codexmedium"]:
                triple_df["relation"] = triple_df["relation_id"].map(
                    relation_id_to_text
                )
            elif DATASET in ["WN18RR", "WN18"]:
                triple_df["relation"] = triple_df["relation_id"].map(
                    data.relation_id_to_label
                )
                triple_df["relation"] = triple_df["relation"].apply(
                    lambda x: x.replace("_", " ")
                )

            sample_df = pd.DataFrame(
                columns=["instruction", "input", "output", "embedding_ids"]
            )
            sample_df["input"] = triple_df.apply(
                lambda row: template[DATASET]["KGED"]["input"].format(
                    row["head"], row["relation"], row["tail"]
                ),
                axis=1,
            )
            sample_df["instruction"] = np.full(
                len(sample_df), template[DATASET]["KGED"]["instruction"]
            )

            if SAMPLE_TYPE == "P":
                sample_df["output"] = np.full(len(sample_df), "Triple Correct")
            elif SAMPLE_TYPE == "N":
                sample_df["output"] = triple_df["neg_type"]
            sample_df["embedding_ids"] = triple_df[
                ["head_id", "relation_id", "tail_id"]
            ].values.tolist()
            return sample_df

        # ======= for trainset ==========
        positive_sample = generate_sample(data=dataset.training, SAMPLE_TYPE="P")
        negtive_sample = generate_sample(data=dataset.training, SAMPLE_TYPE="N")
        sample = pd.concat([positive_sample, negtive_sample])
        sample.to_json(os.path.join(DATA_PATH, "train_4c.json"), orient="records")

        # ======= for testset ==========
        positive_sample = generate_sample(data=dataset.testing, SAMPLE_TYPE="P")
        negtive_sample = generate_sample(data=dataset.testing, SAMPLE_TYPE="N")
        sample = pd.concat([positive_sample, negtive_sample])
        sample.to_json(os.path.join(DATA_PATH, "test_4c.json"), orient="records")

        # ======= for val ==========
        positive_sample = generate_sample(data=dataset.validation, SAMPLE_TYPE="P")
        negtive_sample = generate_sample(data=dataset.validation, SAMPLE_TYPE="N")
        sample = pd.concat([positive_sample, negtive_sample])
        # positive_sample.to_json(os.path.join(DATA_PATH, 'test_triple_positive.json'), orient="records")
        sample.to_json(os.path.join(DATA_PATH, "val_4c.json"), orient="records")

    def get_1000_sample(sample_path):
        dataset_df = pd.read_json(sample_path)
        df_sample = dataset_df.sample(n=1000, random_state=42)
        df_sample.to_json(
            sample_path.replace(".json", "_1000.json"), orient="records", indent=4
        )

    def get_CN(data_path, x):
        """
        data_path: all_edge, including negative samples
        x: node feature
        """
        dataset = get_dataset(dataset=DATASET)
        dataset = dataset.training  # no leakage
        triples = dataset.mapped_triples
        row = triples[:, 0]  # subject
        col = triples[:, 2]  # object
        rel = torch.ones_like(row)
        adj_t = SparseTensor(
            row=row,
            col=col,
            value=rel,
            sparse_sizes=(dataset.num_entities, dataset.num_entities),
        )

        edge_df = pd.read_json(data_path)
        edges_tensor = torch.tensor(edge_df["embedding_ids"])
        edges = [edges_tensor[:, 0], edges_tensor[:, 2]]

        node_label = HoCN()

        (
            count_1_1,
            count_1_2,
            count_2_1,
            count_2_2,
            count_self_1_2,
            count_self_2_1,
            degree_u,
            degree_v,
        ) = node_label.forward(x=x, edges=edges, adj_t=adj_t, node_weight=None)

        triples = list(zip(
            edges_tensor[:, 0].tolist(),
            edges_tensor[:, 1].tolist(),
            edges_tensor[:, 2].tolist(),
        ))

        CN_features = {
            "count_1_1": count_1_1,
            "count_1_2": count_1_2,
            "count_2_1": count_2_1,
            "count_2_2": count_2_2,
            # For some KG may need more self feature
            # "count_self_1_2": count_self_1_2,
            # "count_self_2_1": count_self_2_1,
        }

        # 对每种 count，都建一个 { (h, r, t): feature_row } 的映射
        CN_with_index = {
            name: {triple: feat for triple, feat in zip(triples, tensor)}
            for name, tensor in CN_features.items()
        }

        return CN_with_index

    def pre_compute(feat_type):
        if "KGE_HoA" in feat_type:
            entity_embeddings, relation_embeddings = get_KGE_embed()
            torch.save(entity_embeddings, f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth")
            torch.save(
                relation_embeddings, f"{SAVE_PATH}/{MODEL}/relation_embeddings.pth"
            )
            KGE_HoA_feat = get_high_order_adj(
                feature_path=f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth"
            )
            torch.save(KGE_HoA_feat, f"{SAVE_PATH}/{MODEL}/KGE_HoA_feat.pth")

        elif "SBert_HoA" in feat_type:
            entity_embeddings, relation_embeddings = get_SBert_embed()
            torch.save(entity_embeddings, f"{SAVE_PATH}/Sentence/entity_embeddings.pth")
            torch.save(
                relation_embeddings, f"{SAVE_PATH}/Sentence/relation_embeddings.pth"
            )
            SBert_HoA_feat = get_high_order_adj(
                feature_path=f"{SAVE_PATH}/Sentence/entity_embeddings.pth"
            )
            torch.save(SBert_HoA_feat, f"{SAVE_PATH}/Sentence/SBert_HoA_feat.pth")

        elif "KGE_CN" in feat_type:
            x = torch.load(f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth")
            DATA_PATH = f"{SAVE_PATH}/{MODEL}/train_4c.json"
            KGE_CN = get_CN(data_path=DATA_PATH, x=x)
            torch.save(KGE_CN, f"{SAVE_PATH}/{MODEL}/KGE_CN_feat.pth")

        elif "SBert_CN" in feat_type:
            x = torch.load(f"{SAVE_PATH}/Sentence/entity_embeddings.pth")
            DATA_PATH = f"{SAVE_PATH}/Sentence/train_4c.json"
            SBert_CN = get_CN(data_path=DATA_PATH, x=x)
            torch.save(SBert_CN, f"{SAVE_PATH}/Sentence/SBert_CN_feat.pth")

    def pre_feature(version):
        if version == "1":
            # Only KGE Embedding
            pre_feat = {
                "entity_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth"
                ),
                "relation_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/relation_embeddings.pth"
                ),
            }

        elif version == "2":
            # Only SBert Embedding
            pre_feat = {
                "entity_embed": torch.load(
                    f"{SAVE_PATH}/Sentence/entity_embeddings.pth"
                ),
                "relation_embed": torch.load(
                    f"{SAVE_PATH}/Sentence/relation_embeddings.pth"
                ),
            }

        elif version == "3":
            # KGE + HoA
            KGE_HoA_feat = torch.load(f"{SAVE_PATH}/{MODEL}/KGE_HoA_feat.pth")
            pre_feat = {
                "entity_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth"
                ),
                "relation_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/relation_embeddings.pth"
                ),
                "hop1": KGE_HoA_feat[0],
                "hop2": KGE_HoA_feat[1],
                "hop3": KGE_HoA_feat[2],
            }

        elif version == "4":
            # SBert + HoA
            SBert_HoA_feat = torch.load(f"{SAVE_PATH}/Sentence/SBert_HoA_feat.pth")
            pre_feat = {
                "entity_embed": torch.load(
                    f"{SAVE_PATH}/Sentence/entity_embeddings.pth"
                ),
                "relation_embed": torch.load(
                    f"{SAVE_PATH}/Sentence/relation_embeddings.pth"
                ),
                "hop1": SBert_HoA_feat[0],
                "hop2": SBert_HoA_feat[1],
                "hop3": SBert_HoA_feat[2],
            }

        elif version == "5":
            # KGE + CN
            KGE_CN_feat = torch.load(f"{SAVE_PATH}/{MODEL}/KGE_CN_feat.pth")
            pre_feat = {
                "entity_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/entity_embeddings.pth"
                ),
                "relation_embed": torch.load(
                    f"{SAVE_PATH}/{MODEL}/relation_embeddings.pth"
                ),
                "count_1_1": KGE_CN_feat["count_1_1"],
                "count_1_2": KGE_CN_feat["count_1_2"],
                "count_2_1": KGE_CN_feat["count_2_1"],
                "count_2_2": KGE_CN_feat["count_2_2"],
            }

        torch.save(pre_feat, f"DATA_PATH/pre_feat_v{version}.pth")

    entity_embeddings, relation_embeddings = get_KGE_embed()
    torch.save(entity_embeddings, f"{SAVE_PATH}/RotatE-D512/entity_embeddings.pth")
    torch.save(relation_embeddings, f"{SAVE_PATH}/RotatE-D512/relation_embeddings.pth")

    pre_compute(
        feat_type=[
            "KGE_HoA",
            "SBert_HoA",
            "KGE_CN",
            "SBert_CN",
        ]
    )
    pre_feature(version="1")

    generate_triple_sample()
    get_1000_sample(f"DATA_PATH/TransE-D512/train_4c.json")


if __name__ == "__main__":
    fire.Fire(run)
