# Is It Really That Simple?

Code and data associated with the paper "Is It Really That Simple? Prompting Language Models for Automatic Text Simplification in Italian".

## Getting Started

Beware, the operation might break existing venv/conda environments. We recommend working on a separate environment.
We conducted all our experiments with Python 3.10. To get started, install the requirements listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Run the Simplification

Execute the following command from the root of this project to execute the simplification on Admin-It with Vicuna v1.3 13B.

Notes:
- edit the model name to use different models
- you will likely need to change some of the parameters to adapt to your setup

```bash
./bash/simplify_with_model.sh "lmsys/vicuna-13b-v1.3" "admin-It"
```

If you want to use LoRA weights, then use:
```bash
./bash/simplify_with_lora.sh "yahma/llama-7b-hf" "teelinsan/camoscio-7b-llama" "admin-It"
```

where the first argument is the base pretrained model, and the second is the name of the adapter weights.

### Prompt Template

Both scripts will use the *explicit* prompt template by default. Edit them and pass `--prompt_template="1"` among the input arguments to use the *implicit* template.

