import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
import numpy as np
import pytest
from fastapi.testclient import TestClient
from projects import common, messages
from projects.fixtures import generate

@pytest.fixture(scope="session")
def data(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("data"))

@pytest.fixture(scope="session", autouse=True)
def artifact_root(tmp_path_factory):
    previous = common.ROOT
    common.ROOT = tmp_path_factory.mktemp("models")
    from projects import api
    api.ROOT = common.ROOT
    yield common.ROOT
    common.ROOT = previous

@pytest.fixture(scope="session")
def trained_messages(data):
    return messages.train(data / "messages.csv",epochs=2,synthetic=True)

def test_messages_rnn_reload_preprocessing_and_dual_heads(trained_messages):
    assert messages.preprocess("HI  https://example.com\n Team") == "hi url team"
    model=messages.Classifier()
    result=model.predict("Please track my order")
    assert result["intent"]["label"] in model.config["intents"]
    assert result["sentiment"]["label"] in model.config["sentiments"]
    assert abs(sum(result["intent"]["probabilities"].values())-1) < 1e-5
    with pytest.raises(ValueError):
        model.predict("   ")

def test_groq_missing_key_and_provider_contract(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY",raising=False)
    from projects.api import app
    assert TestClient(app).post("/explain",json={"context":{}}).status_code == 503
    captured={}
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":"Model explanation"}}]}
    def post(self,url,**kwargs):
        captured.update(url=url,**kwargs)
        return FakeResponse()
    monkeypatch.setenv("GROQ_API_KEY","test-key-not-real")
    monkeypatch.setattr(common.httpx.Client,"post",post)
    assert common.groq_explain({"score":.5}) == "Model explanation"
    assert captured["url"].startswith("https://api.groq.com/")
    assert captured["headers"]["Authorization"] == "Bearer test-key-not-real"

def test_home_status_and_invalid_input():
    from projects.api import app
    client=TestClient(app)
    assert client.get("/").status_code==200
    assert "Message Intelligence" in client.get("/").text
    assert client.get("/status").status_code==200
    assert client.post("/messages/classify",json={"text":""}).status_code==422
