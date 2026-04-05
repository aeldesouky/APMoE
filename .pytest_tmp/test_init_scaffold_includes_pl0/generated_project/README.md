# D:\coding_projects\APMoE\.pytest_tmp\test_init_scaffold_includes_pl0\generated_project

An APMoE project for age prediction using Mixture of Experts.

## Quick Start

1. **Configure**: Edit `config.json` to point to your processor, cleaner,
   anonymizer, embedder, and expert implementations.

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
D:\coding_projects\APMoE\.pytest_tmp\test_init_scaffold_includes_pl0\generated_project/
  config.json          # Framework configuration (built-in Keras + ONNX experts)
  custom_expert.py     # Optional: your own ExpertPlugin (default config uses builtins)
  weights/             # face_age_expert.keras, keystroke_*.onnx, keystroke_constants.json
  README.md            # This file
```

## Extending the Framework

- Subclass `ExpertPlugin` in `custom_expert.py`.
- Register your class in `config.json` under `"experts"`.
- See the [APMoE documentation](https://github.com/your-org/apmoe) for details.
