from pyrdf2vec.graphs import KG
import pandas as pd
import rdflib as rdflib

# Load CSV file with country names and their labels
country_data = pd.read_csv('countries.csv')
entities = country_data['Country']

# We will exclude triples (s, p, o) with p in label_predicates from our KG
# as these do not carry any useful information.
label_predicates = [
     'http://dbpedia.org/ontology/abstract',
     'http://dbpedia.org/ontology/flag',
     'http://dbpedia.org/ontology/thumbnail',
     'http://dbpedia.org/ontology/wikiPageExternalLink',
     'http://dbpedia.org/ontology/wikiPageID',
     'http://dbpedia.org/ontology/wikiPageRevisionID',
     'http://dbpedia.org/ontology/wikiPageWikiLink',
     'http://dbpedia.org/property/flagCaption',
     'http://dbpedia.org/property/float',
     'http://dbpedia.org/property/footnoteA',
     'http://dbpedia.org/property/footnoteB',
     'http://dbpedia.org/property/footnoteC',
     'http://dbpedia.org/property/source',
     'http://dbpedia.org/property/width',
     'http://purl.org/dc/terms/subject',
     'http://purl.org/linguistics/gold/hypernym',
     'http://purl.org/voc/vrank#hasRank',
     'http://www.georss.org/georss/point',
     'http://www.w3.org/2000/01/rdf-schema#comment',
     'http://www.w3.org/2000/01/rdf-schema#label',
     'http://www.w3.org/2000/01/rdf-schema#seeAlso',
     'http://www.w3.org/2002/07/owl#sameAs',
     'http://www.w3.org/2003/01/geo/wgs84_pos#geometry',
     'http://dbpedia.org/ontology/wikiPageRedirects',
     'http://www.w3.org/2003/01/geo/wgs84_pos#lat',
     'http://www.w3.org/2003/01/geo/wgs84_pos#long',
     'http://www.w3.org/2004/02/skos/core#exactMatch',
     'http://www.w3.org/ns/prov#wasDerivedFrom',
     'http://xmlns.com/foaf/0.1/depiction',
     'http://xmlns.com/foaf/0.1/homepage',
     'http://xmlns.com/foaf/0.1/isPrimaryTopicOf',
     'http://xmlns.com/foaf/0.1/name',
     'http://dbpedia.org/property/website',
     'http://dbpedia.org/property/west',
     'http://dbpedia.org/property/wordnet_type',
     'http://www.w3.org/2002/07/owl#differentFrom',
]

# KG Loading Alternative 1: Loading the entire turtle file into memory
kg = KG("countries.ttl", file_type='turtle',
        label_predicates=[rdflib.URIRef(x) for x in label_predicates])
# KG Loading Alternative 2: Using a dbpedia endpoint (nothing is loaded into memory)
kg = KG("https://dbpedia.org/sparql", is_remote=True,
        label_predicates=[rdflib.URIRef(x) for x in label_predicates])

# Make sure that every entity can be found in our KG
filtered_entities = [e for e in entities if e in kg._entities]
not_found = set(entities) - set(filtered_entities)
print(f'{not_found} could not be found in the KG! Removing them...')
entities = filtered_entities

from pyrdf2vec import RDF2VecTransformer
import numpy as np

transformer = RDF2VecTransformer()
transformer.fit(kg, entities, verbose=True)
walk_embeddings = transformer.transform(entities)
print(len(entities), np.array(walk_embeddings).shape)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

walk_tsne = TSNE(random_state=42)
X_tsne = walk_tsne.fit_transform(walk_embeddings)

plt.figure(figsize=(15, 15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

texts = []
for x, y, lab in zip(X_tsne[:, 0], X_tsne[:, 1], entities):
    lab = lab.split('/')[-1]
    text = plt.text(x, y, lab)
    texts.append(text)
    
adjust_text(texts, lim=5, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()