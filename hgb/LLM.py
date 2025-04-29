import torch

import os
from openai import OpenAI


client = OpenAI(base_url = 'https://api.nextapi.fun', api_key = 'ak-7FLJwZX9hiTVTEoCBMrpFeFZbMJKOMoyykmXqr4A2doAxgPp')


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

    def graph_encode(self):
        if self.dataset == "Aminer":
            text = "Aminer is a temporal heterogeneous graph dataset about academic citations. Its time slices are separated using the publication year (during 1990-2006) of papers. The graph consists of three types of nodes (paper, author and venue), and two types of relations (paper-publish-venue and author-writer-paper). "
            
        elif self.dataset == "Ecomm":
            text = "Ecomm is a real-world temporal heterogeneous bipartite graph of the ecommerce, which mainly records shopping behaviors of users within 11 daily snapshots from 10th June 2019 to 20th June 2019. It consists of two types of nodes (user and item) and four types of relations (user-click-item, user-buy-item, user-(add-to-cart)-item and user-(add-to-favorite)-item)."
        elif self.dataset == "IMDB":
            text = "IMDB is a huge data resource based on the Internet Movie Database (IMDb). It consists of three types of nodes (movie, director, actor) and two types of relationships (director-creation-movie, actor-release-movie)."
        elif self.dataset == "ACM":
            text = "The ACM dataset covers research papers, conference proceedings, technical reports, and book chapters in computer science and related fields from the 1950s to the present. It consists of three types of nodes (paper, author and subject) and three types of relationships (author-publish-paper, paper-cite-paper, paper-belongs-subject)."
        elif self.dataset == "DBLP":
            text = "The construction of the DBLP dataset is based on academic literature in the field of computer science, covering a wide range of research topics. It consists of four types of nodes (author, paper, term, venue) and three types of relationships (author-publishes-paper, paper-contains-term, paper-belongs-venue)."
        elif self.dataset == "Yelp-nc":
            text = "Yelp is a business review net containing timestamped user reviews and tips on businesses. There are two types of nodes (users and business) and two types of edges (user-tip-business and userreview-business) in the temporal heterogeneous graph constructed based on it."
        request = "Please output a summary of the information about this heterogeneous graph in the following format: {NodeType:,Attribute:}."
        summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text+request}
            ]
        ).choices[0].message.content
        emb = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=summary
                ).data[0].embedding

        emb = torch.tensor(emb)
        emb = emb.to(self.device)
        return emb
