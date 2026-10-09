"""Shared embeddings with LSTM/GRU branches and intent/sentiment heads."""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from .common import directory, metadata

def preprocess(text):
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", " url ", str(text).lower())).strip()

def train(csv, epochs=10, synthetic=False):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    data = pd.read_csv(csv).dropna(subset=["text", "intent", "sentiment"])
    data.text = data.text.map(preprocess)
    if len(data) < 30 or data.intent.value_counts().min() < 4 or (data.text.str.len() == 0).any():
        raise ValueError("Need 30 nonempty labeled messages, at least four per intent.")
    intents, sentiments = sorted(data.intent.unique().tolist()), sorted(data.sentiment.unique().tolist())
    if len(intents) < 2 or len(sentiments) < 2:
        raise ValueError("Need multiple intent and sentiment classes.")
    tr, te = train_test_split(np.arange(len(data)), stratify=data.intent, test_size=.25, random_state=42)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=8000, output_sequence_length=60)
    vectorizer.adapt(tf.constant(data.text.iloc[tr].tolist()))
    tokens = vectorizer(tf.constant(data.text.tolist())).numpy()
    inputs = tf.keras.Input((60,), dtype="int32")
    embedded = tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 32, mask_zero=True)(inputs)
    lstm = tf.keras.layers.LSTM(24)(embedded)
    gru = tf.keras.layers.GRU(24)(embedded)
    shared = tf.keras.layers.Concatenate()([lstm, gru])
    model = tf.keras.Model(inputs, {"intent": tf.keras.layers.Dense(len(intents), activation="softmax", name="intent")(shared), "sentiment": tf.keras.layers.Dense(len(sentiments), activation="softmax", name="sentiment")(shared)})
    model.compile(optimizer="adam", loss={"intent": "sparse_categorical_crossentropy", "sentiment": "sparse_categorical_crossentropy"})
    targets = {"intent": np.asarray([intents.index(i) for i in data.intent]), "sentiment": np.asarray([sentiments.index(s) for s in data.sentiment])}
    model.fit(tokens[tr], {k: v[tr] for k,v in targets.items()}, epochs=epochs, verbose=0)
    pred = model(tokens[te], training=False)
    report = {"synthetic": synthetic, "held_out_rows": len(te), "metrics": {k: {"accuracy": float(accuracy_score(targets[k][te], pred[k].numpy().argmax(1))), "macro_f1": float(f1_score(targets[k][te], pred[k].numpy().argmax(1), average="macro", zero_division=0))} for k in targets}}
    path = directory("messages")
    model.save(path / "model.keras")
    metadata(path / "vocabulary.json", {"words": vectorizer.get_vocabulary(), "intents": intents, "sentiments": sentiments})
    metadata(path / "report.json", report)
    return report

class Classifier:
    def __init__(self):
        import tensorflow as tf
        path = directory("messages")
        self.config = metadata(path / "vocabulary.json")
        self.model = tf.keras.models.load_model(path / "model.keras", compile=False)
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=8000, output_sequence_length=60, vocabulary=self.config["words"][2:])
        self.report = metadata(path / "report.json")

    def predict(self, text):
        import tensorflow as tf
        clean = preprocess(text)
        if not clean or len(clean) > 4000:
            raise ValueError("Message must contain 1–4000 characters.")
        output = self.model(self.vectorizer(tf.constant([clean])), training=False)
        result = {"normalized": clean, "evaluation": self.report}
        for head, classes in [("intent", self.config["intents"]), ("sentiment", self.config["sentiments"])]:
            probs = output[head].numpy()[0]
            result[head] = {"label": classes[int(probs.argmax())], "confidence": float(probs.max()), "probabilities": dict(zip(classes, map(float, probs)))}
        result["review_required"] = result["intent"]["confidence"] < .6 or self.report["synthetic"]
        return result
