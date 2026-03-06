```mermaid
flowchart TB

    %% --- Entry ---
    User["User Input"]

    %% --- Modality Factory ---
    User --> ModalityFactory["Modality Processor Factory"]

    %% --- Modality Processors (Strategy Pattern) ---
    subgraph ModalityLayer ["Modality Processing Layer (parallel)"]
        VProc["Visual Processor"]
        AProc["Audio Processor"]
        EProc["EEG Processor"]
    end

    ModalityFactory --> VProc
    ModalityFactory --> AProc
    ModalityFactory --> EProc

    %% --- Internal Pipeline Per Modality ---
    subgraph ProcessorPipeline ["Processor Internal Pipeline (Pipeline Pattern)"]
        Clean["Cleaner Strategy"]
        Anon["Anonymizer Strategy"]
        Embed["Embedding Strategy (optional)"]
    end

    VProc --> Clean
    AProc --> Clean
    EProc --> Clean

    Clean --> Anon
    Anon -->|"always"| Embed
    Anon -->|"if no embedder configured"| DataMap

    %% --- Processed Data Map ---
    Embed -->|"EmbeddingResult"| DataMap["Processed Data Map: modality name to embeddings or preprocessed data"]

    %% --- Expert Registry ---
    DataMap --> ExpertRegistry["Expert Registry (dispatches by declared modalities)"]

    %% --- Experts (Plugin Pattern) ---
    subgraph ExpertLayer ["Expert Plugins (pretrained, one or more modalities each)"]
        Expert1["Expert Plugin 1 (e.g. expects embeddings)"]
        Expert2["Expert Plugin 2 (e.g. expects preprocessed data)"]
        ExpertN["Expert Plugin N (e.g. multi-modal, mixed)"]
    end

    ExpertRegistry -->|"subset of processed data"| Expert1
    ExpertRegistry -->|"subset of processed data"| Expert2
    ExpertRegistry -->|"subset of processed data"| ExpertN

    %% --- Aggregation ---
    Expert1 -->|"age prediction + confidence"| Aggregator["Aggregation Strategy (math formula or small model)"]
    Expert2 -->|"age prediction + confidence"| Aggregator
    ExpertN -->|"age prediction + confidence"| Aggregator

    %% --- Final Output ---
    Aggregator --> Output["Final Age Prediction"]
```
