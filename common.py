import torch
import pandas as pd

from utils.nlp_utils import NLPUtils


class DocumentReport:
    def __init__(self, app_id):
        self.app_id = app_id
        self.permissions = {}
        self.preprocessed_sentences = []
        self.sentences = []
        self.prediction_result = None
        self.index_tensors = None


class SentenceReport:
    def __init__(self, id, sentence):
        self.app_id = id
        self.sentence = sentence
        self.permissions = {}
        self.preprocessed_sentence = None
        self.prediction_result = None
        self.index_tensor = None


class Review:
    def __init__(self, sentence, score):
        self.sentence = sentence
        self.preprocessed_sentence = None
        self.score = score
        self.index_tensor = None
        self.prediction_result = None


def load_row_document_acnet_file(infile, stemmer, embeddings, filtered_words):
    print("Loading row {} ".format(infile))
    # read training data
    print("Reading Train Sentences")
    tagged_train_file = pd.read_csv(infile)
    documents = []
    acnet_map = {
        "RECORD_AUDIO": "MICROPHONE",
        "READ_CONTACTS": "CONTACTS",
        "READ_CALENDAR": "CALENDAR",
        "ACCESS_FINE_LOCATION": "LOCATION",
        "CAMERA": "CAMERA",
        "READ_SMS": "SMS",
        "READ_CALL_LOGS": "CALL_LOG",
        "CALL_PHONE": "PHONE",
        "WRITE_SETTINGS": "SETTINGS",
        "GET_TASKS": "TASKS",
        "STORAGE": "STORAGE",
    }

    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]

        if documents == []:  # if it is the first document
            documents.append(DocumentReport(app_id))
        elif documents[-1].app_id != app_id:  # if it is a new document
            documents.append(DocumentReport(app_id))

        for permission in acnet_map:
            if (
                permission not in documents[-1].permissions
                or row[acnet_map[permission]] == 1
            ):
                documents[-1].permissions[permission] = row[acnet_map[permission]]

        documents[-1].sentences.append(sentence)
        preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)

        filtered = []
        for word in preprocessed:
            if word in embeddings and word in filtered_words:
                filtered.append(word)
        documents[-1].preprocessed_sentences.append(filtered)

    print("Loading completed")
    return documents


def load_row_sentence_acnet_file(infile, stemmer, embeddings):
    print("Loading row {} ".format(infile))
    # read training data
    print("Reading Train Sentences")
    tagged_train_file = pd.read_csv(infile)
    train_sententence_reports = []
    acnet_map = {
        "RECORD_AUDIO": "MICROPHONE",
        "READ_CONTACTS": "CONTACTS",
        "READ_CALENDAR": "CALENDAR",
        "ACCESS_FINE_LOCATION": "LOCATION",
        "CAMERA": "CAMERA",
        "READ_SMS": "SMS",
        "READ_CALL_LOGS": "CALL_LOG",
        "CALL_PHONE": "PHONE",
        "WRITE_SETTINGS": "SETTINGS",
        "GET_TASKS": "TASKS",
        "STORAGE": "STORAGE",
    }
    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]
        sentence_report = SentenceReport(app_id, sentence)

        for permission in acnet_map:
            sentence_report.permissions[permission] = row[acnet_map[permission]]

        preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
        sentence_report.preprocessed_sentence = [
            word for word in preprocessed if word in embeddings
        ]
        if sentence_report.preprocessed_sentence != []:
            train_sententence_reports.append(sentence_report)
    print("Loading completed")
    return train_sententence_reports


def load_row_reviews(infile, stemmer, embeddings):
    print("Loading row {} ".format(infile))
    reviews = {}
    tagged_train_file = pd.read_csv(infile)
    for idx, row in tagged_train_file.iterrows():
        if idx != 0 and idx % 1000 == 0:
            print(idx)
        app_id, sentence, score = (
            row["application_id"],
            row["review_sentence"],
            row["score"],
        )
        if app_id and sentence and score:
            preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
            if len(preprocessed) != 0:
                review = Review(sentence, score)
                if app_id not in reviews:
                    reviews[app_id] = []
                review.preprocessed_sentence = [
                    word for word in preprocessed if word in embeddings
                ]
                reviews[app_id].append(review)
    return reviews


def load_embeddings(options):
    if options.external_embedding is not None:
        if os.path.isfile(
            os.path.join(options.saved_parameters_dir, options.saved_prevectors)
        ):
            ext_embeddings, _ = IOUtils.load_embeddings_file(
                os.path.join(options.saved_parameters_dir, options.saved_prevectors),
                "pickle",
                options.lower,
            )
            return ext_embeddings
        else:
            ext_embeddings, _ = IOUtils.load_embeddings_file(
                options.external_embedding,
                options.external_embedding_type,
                options.lower,
            )
            IOUtils.save_embeddings(
                os.path.join(options.saved_parameters_dir, options.saved_prevectors),
                ext_embeddings,
            )
            return ext_embeddings
    else:
        raise Exception("external_embedding option is None")


def create_document_index_tensors(documents, w2i):
    def get_tensor(sequence, w2i):
        index_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        for idx, word in enumerate(sequence):
            index_tensor[0][idx] = w2i[word]
        return index_tensor

    for document in documents:
        document.index_tensors = []
        for preprocessed_sentence in document.preprocessed_sentences:
            index_tensor = get_tensor(preprocessed_sentence, w2i)
            document.index_tensors.append(index_tensor)


def create_sentence_index_tensors(sentences, w2i):
    def get_tensor(sequence, w2i):
        index_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        for idx, word in enumerate(sequence):
            index_tensor[0][idx] = w2i[word]
        return index_tensor

    for sentence in sentences:
        sentence.index_tensor = get_tensor(sentence.preprocessed_sentence, w2i)


def calculate_freqs(infile, stemmer, embeddings):
    tagged_train_file = pd.read_csv(infile)
    vocab_freq = {}
    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]

        preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
        for token in preprocessed:
            if token not in vocab_freq:
                vocab_freq[token] = 0
            vocab_freq[token] += 1
    return vocab_freq
