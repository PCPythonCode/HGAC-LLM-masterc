import torch

import os
from openai import OpenAI

class LLM_model(torch.nn.Module):
    def __init__(
        self,
        hidden,
        dataset,
        device
    ):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.graph_data = dataset.dataset

    def _build_path_context_Ecomm(self, target_user_id, max_paths=4):
        context_list = []
        path_count = 0
        for edge in self.graph_data["edges"]:
            src, dst, edge_type = edge
            if src == target_user_id and edge_type in ["buys", "clicks", "add-to-cart", "add-to-favorite"]:
                item_attr = self.graph_data["item"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in item_attr.items() if v]) 
                context_list.append(f"Item {dst} -> {attr_text}")
                path_count += 1

            if path_count >= max_paths:
                break
        return " ".join(context_list) if context_list else "No attribute context found."
    
    def _build_path_context_dblp(self, target_node_id, max_paths=4):
        context_list = []
        path_count = 0
        edges = self.graph_data["edges"]

        for edge in edges:
            src, dst, edge_type = edge
            if src == target_node_id and edge_type == "author-publishes-paper":
                paper_attr = self.graph_data["paper"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in paper_attr.items() if v])
                context_list.append(f"Author {src} published Paper {dst} with {attr_text}")
                path_count += 1
            elif src == target_node_id and edge_type == "paper-contains-term":
                term_attr = self.graph_data["term"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in term_attr.items() if v])
                context_list.append(f"Paper {src} contains Term {dst} with {attr_text}")
                path_count += 1
            elif src == target_node_id and edge_type == "paper-belongs-venue":
                venue_attr = self.graph_data["venue"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in venue_attr.items() if v])
                context_list.append(f"Paper {src} published at Venue {dst} with {attr_text}")
                path_count += 1
            if path_count >= max_paths:
                break

        return " ".join(context_list) if context_list else "No attribute context found."
    
    def _build_path_context_acm(self, target_paper_id, max_paths=4):
        context_list = []
        path_count = 0
        edges = self.graph_data["edges"]

        for edge in edges:
            src, dst, edge_type = edge
            if src == target_paper_id and edge_type == "paper-cite-paper":
                cited_paper_attr = self.graph_data["paper"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in cited_paper_attr.items() if v])
                context_list.append(f"Paper {src} cites Paper {dst} with {attr_text}")
                path_count += 1
            elif dst == target_paper_id and edge_type == "author-publish-paper":
                author_attr = self.graph_data["author"].get(src, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in author_attr.items() if v])
                context_list.append(f"Paper {dst} is written by Author {src} with {attr_text}")
                path_count += 1
            elif src == target_paper_id and edge_type == "paper-belongs-subject":
                subject_attr = self.graph_data["subject"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in subject_attr.items() if v])
                context_list.append(f"Paper {src} belongs to Subject {dst} with {attr_text}")
                path_count += 1
            if path_count >= max_paths:
                break
        return " ".join(context_list) if context_list else "No attribute context found."
    def _build_path_context_imdb(self, target_movie_id, max_paths=4):
        context_list = []
        path_count = 0
        edges = self.graph_data["edges"]

        for edge in edges:
            src, dst, edge_type = edge
            if src == target_movie_id and edge_type == "movie-directed-director":
                director_attr = self.graph_data["director"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in director_attr.items() if v])
                context_list.append(f"Movie {src} was directed by Director {dst} with {attr_text}")
                path_count += 1
            elif src == target_movie_id and edge_type == "movie-acted-actor":
                actor_attr = self.graph_data["actor"].get(dst, {})
                attr_text = " ".join([f"{k}: {v}" for k, v in actor_attr.items() if v])
                context_list.append(f"Movie {src} features Actor {dst} with {attr_text}")
                path_count += 1

            if path_count >= max_paths:
                break

        return " ".join(context_list) if context_list else "No attribute context found."
    def _build_path_context_yelp(self, target_node_id, max_paths=4):
        context_list = []
        path_count = 0
        for src, dst, edge_type in self.graph_data["edges"]:
            if src == target_node_id and edge_type in ["user-review-business", "user-tip-business"]:
                business_attr = self.graph_data["business"].get(dst, {})
                attr_text = " ".join(f"{k}: {v}" for k, v in business_attr.items() if v)
                action = "reviewed" if edge_type == "user-review-business" else "tipped"
                context_list.append(f"User {src} {action} Business {dst} with {attr_text}")
                path_count += 1
            elif src == target_node_id and edge_type == "business-provides-service":
                service_attr = self.graph_data["service"].get(dst, {})
                attr_text = " ".join(f"{k}: {v}" for k, v in service_attr.items() if v)
                context_list.append(f"Business {src} provides Service {dst} with {attr_text}")
                path_count += 1
            elif src == target_node_id and edge_type == "business-has-level":
                level_attr = self.graph_data["level"].get(dst, {})
                attr_text = " ".join(f"{k}: {v}" for k, v in level_attr.items() if v)
                context_list.append(f"Business {src} has Level {dst} with {attr_text}")
                path_count += 1

            if path_count >= max_paths:
                break
        return " ".join(context_list) if context_list else "No attribute context found."
    def _build_path_context_aminer(self, target_node_id, max_paths=4):
        context_list = []
        path_count = 0
        for src, dst, edge_type in self.graph_data["edges"]:
            if src == target_node_id and edge_type == "paper-publish-venue":
                venue_attr = self.graph_data["venue"].get(dst, {})
                attr_text = " ".join(f"{k}: {v}" for k, v in venue_attr.items() if v)
                context_list.append(f"Paper {src} published in Venue {dst} with {attr_text}")
                path_count += 1
            elif src == target_node_id and edge_type == "author-writer-paper":
                paper_attr = self.graph_data["paper"].get(dst, {})
                attr_text = " ".join(f"{k}: {v}" for k, v in paper_attr.items() if v)
                context_list.append(f"Author {src} wrote Paper {dst} with {attr_text}")
                path_count += 1
            if path_count >= max_paths:
                break

        return " ".join(context_list) if context_list else "No attribute context found."
    def graph_encode(self):
        if self.dataset == "Aminer":
            missing_node_ids = []
            for node_id, attr in self.graph_data["author","venue"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_aminer(missing_node_ids)            
        elif self.dataset == "Ecomm":
            missing_node_ids = []
            for node_id, attr in self.graph_data["item"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_Ecomm(missing_node_ids)
        elif self.dataset == "Yelp-nc":
            missing_node_ids = []
            for node_id, attr in self.graph_data["user","service","level"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_yelp(missing_node_ids) 
        elif self.dataset == "IMDB":
            missing_node_ids = []
            for node_id, attr in self.graph_data["director","actor"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_imdb(missing_node_ids) 
        elif self.dataset == "DBLP":
            missing_node_ids = []
            for node_id, attr in self.graph_data["author","term","venue"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_dblp(missing_node_ids) 
        elif self.dataset == "ACM":
            missing_node_ids = []
            for node_id, attr in self.graph_data["author","subject"].items():
                if any(v == "" or v is None for v in attr.values()):
                    missing_node_ids.append(node_id)
            text = self._build_path_context_acm(missing_node_ids) 
        request = "Please output a summary of the information about this heterogeneous graph in the following format: {NodeType:,Attribute:}."
        print(request)
        summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a good assistant. Please infer the information of nodes without attributes based on the known information of each node."},
                {"role": "user", "content": text+request}
            ]
        ).choices[0].message.content
        # print(summary)
        emb = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=summary
                ).data[0].embedding#["data"][0]["embedding"]
        #print(emb.data[0].embedding)

        emb = torch.tensor(emb)
        emb = emb.to(self.device)
        return emb
    
    
