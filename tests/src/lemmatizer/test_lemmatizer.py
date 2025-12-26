import pytest
from lemmatizer.lemmatizer import Lemmatizer, TokenLemma

service = Lemmatizer(use_gpu=False)

@pytest.mark.parametrize("word,expected", [
    ("блохи", "блоха"),
    ("замка", "замок"),
    ("пʼятницю", "пʼятниця"),
    ("обʼєктів", "обʼєкт"),
    ("класико-романтична", "класико-романтичний"),
    ("водопою", "водопій"),
    ("червоногарячого", "червоногарячий"),
    ("жовто-блакитного", "жовто-блакитний"),
])
def test_get_lemma(word, expected):
    assert service.get_lemma(word) == expected

def test_analyze_sentence():
    sent = "Блохи кусають собаку у замку у пʼятницю."
    result = service.analyze_sentence(sent)
    assert isinstance(result, list)
    assert any(t.lemma == "блоха" for t in result)
    assert any(t.lemma == "замок" for t in result)
    assert any(t.lemma == "пʼятниця" for t in result)
    for token in result:
        assert isinstance(token, TokenLemma)
        assert isinstance(token.word, str)
        assert isinstance(token.lemma, str)
        assert isinstance(token.pos, str)
