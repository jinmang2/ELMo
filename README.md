ELMo: Embeddings from Language Model
======

Feature-based pre-training Language Model

#### 20.04.24
- 기존에 작업하던 `elmo`파일을 `old_`로 이전
- 새롭게 `elmo`파일을 작성, 앞으로 작성할 내용을 기입
    - charCNN의 논문에서 언급하는 바와 같이 unicode로 변환하여 단어를 vocab으로 만드는 코드 작성
        - 생각해보니, `WordEmbeddingLayer`은 그냥 `EmbeddingLayer`를 상속받게 만들고
        - `CharEmbeddingLayer`에서 dict, emb가 인자로 들어오지 않은 경우 정의할 `to_unicode` 함수로 처리하게 만들면 되지 않나? 오호?
    - vocab의 token을 처리하는 추상화 클래스 작성
    - pre-train과 elmo-layer 적용 부분을 모델이 달리할 수 있도록 코드 작성
    - 목표는
        - NLU model `ELMo`/`BERT`, NLG 모델 `GPT` 등을 내가 정확히 이해하고
        - 최신의 `T5`, `ELECTRA`, `XlNet` 등도 파악,
        - 자연어 처리의 다양한 task를 푸는 것!
