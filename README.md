# OmniGen

A flexible benchmark/data generation framework.

## Quick Start

### 1. Installation

```bash
git clone <https://github.com/little1d/OmniGen>
cd OmniGen
pip install -e .
```

### 2. Configuration

Create a configuration file (e.g., `config.yaml`):

```yaml
dataset_path: "username/dataset-name"
num_samples: 10
output_path: "output/benchmark.json"

generator_config:
  question_types: ["type_cls"]
  prompt_templates:
    type_cls:
      template: "What type of data is this? {available_data}"
      num_choices: 4
    identification:
      template: "Identify the main feature in this data: {available_data}"
      num_choices: 4
  loader_kwargs:
    max_samples: 100

model_config:
  model_type: "internvl"
  api_key: "${INTERNVL_API_KEY}"
  base_url: "${INTERNVL_BASE_URL}"
  model_name: "${INTERNVL_MODEL_NAME}"
```

### 3. Environment Setup

Create a `.env` file:

```env
INTERNVL_API_KEY=your_api_key_here
INTERNVL_BASE_URL=https://api.example.com
INTERNVL_MODEL_NAME=internvl2.5
```

### 4. Generate Benchmarks

```bash
# Basic usage
python generate_benchmark.py --config config.yaml

# Override parameters
python generate_benchmark.py \
  --config config.yaml \
  --question-type type_cls \
  --samples 5 \
  --data-source "username/dataset-name" \
  --max-data-samples 50 \
  --output "benchmark.json"
```

## Output Format

Generated benchmarks are saved as JSON with the following structure:

```json
[
  {
    "question": "What type of spectrum is this?",
    "choices": [
      "Infrared Spectrum (IR)",
      "Proton Nuclear Magnetic Resonance (H-NMR)",
      "Mass Spectrometry (MS)",
      "Carbon-13 Nuclear Magnetic Resonance (C-NMR)"
    ],
    "answer": "Proton Nuclear Magnetic Resonance (H-NMR)",
    "images": {
      "H-NMR": "output/temp_images/sample_0_h_nmr.png"
    }
  }
]
```
