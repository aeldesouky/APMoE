# APMoE System Architecture

This diagram illustrates the core components, data flow, and Inversion of Control (IoC) boundaries of the **Age Prediction Mixture of Experts (APMoE)** framework.

```mermaid
graph TD
    %% Define styles
    classDef interface fill:#e1f5fe,stroke:#3182ce,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#0288d1,stroke-width:2px;
    classDef pipeline fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef expert fill:#fce4ec,stroke:#f57c00,stroke-width:2px;
    classDef external fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% 1. User Interfaces
    subgraph Interfaces ["User / System Interfaces"]
        CLI["CLI (apmoe init, validate, predict)"]:::interface
        API["FastAPI HTTP Server (apmoe serve)"]:::interface
        MW["Middleware (Auth, Rate Limit, CORS)"]:::interface
        API --> MW
    end

    %% 2. Configuration & Core App
    subgraph AppCore ["IoC Container & Core (APMoEApp)"]
        Config[("config.json\n(Defines structure & paths)")]:::external
        Registries["Registries / Dependency Injection"]:::core
        Config -.->|Loads & Resolves| Registries
    end

    %% 3. Inference Pipeline execution
    subgraph Flow ["Inference Pipeline"]
        RawData["Raw Multimodal Bytes / JSON"]:::core
        
        %% Modality Branch Example
        subgraph ModalityBranch ["Modality Chain (e.g., Image / Keystroke)"]
            Proc["ModalityProcessor\n(Extracts ModalityData)"]:::pipeline
            Clean["CleanerStrategy\n(Sanitize)"]:::pipeline
            Anon["AnonymizerStrategy\n(Privacy/Obfuscation)"]:::pipeline
            Embed["EmbedderStrategy\n(Optional Features)"]:::pipeline
            
            Proc --> Clean --> Anon -.-> Embed
        end
        
        RawData --> Proc
    end

    %% 4. Mixture of Experts Component
    subgraph MoE ["Mixture of Experts"]
        Exp1["ExpertPlugin\n(e.g., FaceAgeExpert - Keras)"]:::expert
        Exp2["ExpertPlugin\n(e.g., KeystrokeAgeExpert - ONNX)"]:::expert
        ExpCustom["ExpertPlugin\n(Third-party Custom)"]:::expert
    end

    %% 5. Aggregation
    subgraph Consensus ["Aggregation"]
        Agg["AggregatorStrategy\n(e.g., WeightedAverage)"]:::pipeline
        Pred["Prediction Result\n(Age, Confidence, Metadata)"]:::core
    end

    %% Wiring everything together
    CLI --> |Sends Data| RawData
    MW --> |Sends Data| RawData
    
    Registries -.-> |Instantiates| ModalityBranch
    Registries -.-> |Initializes| MoE

    Anon --> Exp1
    Embed --> Exp1
    Anon --> Exp2
    Anon --> ExpCustom

    Exp1 -->|ExpertOutput| Agg
    Exp2 -->|ExpertOutput| Agg
    ExpCustom -->|ExpertOutput| Agg
    
    Agg --> Pred
    Pred --> |Returned as JSON| CLI
    Pred --> |HTTP Response| API
```

## Layers Overview

1. **Interfaces**: The entry points for using the generic framework. The framework comes with full-featured CLI scaffolding tools and a dynamic FastAPI server equipped with web security middleware.
2. **IoC Container (`APMoEApp`)**: Acts as the brain of the framework. It reads `config.json` via Pydantic and dynamically invokes the specified custom behaviors stored in the Registries without needing hard-coded imports.
3. **Modality Chains**: The data-preparation pipeline segment dynamically generated for each referenced modality (like `image` or `keystroke`). Converts binary inputs into strict `ModalityData` records and passes them through registered Cleaner and Anonymizer algorithms.
4. **Mixture of Experts**: The core predictor plugins. They declare what modalities they require, receive the prepared data (from one or multiple modalities), run their underlying ML logic (e.g. Keras, ONNX, PyTorch), and output decoupled age determinations.
5. **Aggregation**: Gathers the array of disparate predictions and uses configured heuristics (e.g., variance bounds, confidence weights) to compute the final, serialized `Prediction` dataclass sent back to the user interface.