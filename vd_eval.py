import json
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_db(axis):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_database = Chroma(persist_directory=f"./{axis}_db", embedding_function=embedding_model)
    return vector_database

def classify_descriptor(descriptor, threshold, vector_database, K):

    # Search for a similar sentence
    results = vector_database.similarity_search_with_score(descriptor, k=K)

    # filtering out indicators with similarity lower than threshold
    results_filtered = [indi for indi in results if (1 - indi[1]) > threshold]

    # Axis score
    sim_scores = [1 - indi[1] for indi in results_filtered]
    class_scores = [indi[0].metadata['score'] for indi in results_filtered]
    axis_score = sum(sim_scores[i] * class_scores[i] for i in range(len(sim_scores))) / sum(sim_scores)

    return axis_score

def eval(axis):

    with open(f"descriptors_70B_{axis}.json", "r", encoding="utf-8") as file:
        descriptors = json.load(file)

    vector_database = setup_db(axis)

    for descriptor in tqdm(descriptors):
        descriptor_text = descriptor['descriptor']
        axis_score = classify_descriptor(descriptor_text, 0.5, vector_database)
        descriptor['axis_score'] = axis_score

    with open(f"descriptors_70B_{axis}_results.json", "w", encoding="utf-8") as file:
        json.dump(descriptors, file, indent=4, ensure_ascii=False)

    true_labels = [descriptor['label'] for descriptor in descriptors]
    predicted_scores = [descriptor['axis_score'] for descriptor in descriptors]
    predicted_labels = [1 if score > 0 else 0 for score in predicted_scores]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")