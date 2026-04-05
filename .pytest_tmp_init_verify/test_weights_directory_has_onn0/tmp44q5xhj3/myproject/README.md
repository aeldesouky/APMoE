# myproject

An APMoE project for age prediction using Mixture of Experts.

## Quick Start

1. **Configure**: Edit `config.json` to point to your processor, cleaner,
   anonymizer, embedder, expert, and aggregator implementations.

2. **Weights**: Default models are already under `weights/`. Replace or add
   files there if you train your own checkpoints (and update paths in `config.json`).

3. **Validate**: Check the configuration is correct:
   ```
   apmoe validate --config config.json
   ```

4. **Serve**: Start the HTTP API:
   ```
   apmoe serve --config config.json
   ```

5. **Predict**: Run inference on local files:
   ```
   apmoe predict --config config.json --input data/
   ```

## Project Structure

```
myproject/
  config.json          # Framework configuration (built-in Keras + ONNX experts)
  custom_processor.py  # Optional: your own ModalityProcessor stubs
  custom_cleaner.py    # Optional: your own CleanerStrategy stubs
  custom_anonymizer.py # Optional: your own AnonymizerStrategy stubs
  custom_embedder.py   # Optional: your own EmbedderStrategy stubs
  custom_expert.py     # Optional: your own ExpertPlugin (default config uses builtins)
  custom_aggregator.py # Optional: your own AggregatorStrategy stubs
  weights/             # face_age_expert.keras, keystroke_*.onnx, keystroke_constants.json
  README.md            # This file
```

## Extending the Framework

- Subclass `ModalityProcessor` in `custom_processor.py`.
- Subclass `CleanerStrategy` in `custom_cleaner.py`.
- Subclass `AnonymizerStrategy` in `custom_anonymizer.py`.
- Subclass `EmbedderStrategy` in `custom_embedder.py` when you need embeddings.
- Subclass `ExpertPlugin` in `custom_expert.py`.
- Subclass `AggregatorStrategy` in `custom_aggregator.py`.
- Reference your custom classes in `config.json` with dotted paths like `"custom_expert.MyCustomExpert"`.
- See the [APMoE documentation](https://github.com/your-org/apmoe) for details.
