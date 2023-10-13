from transformers import HfArgumentParser
import pandas as pd
from simple_generation import DefaultGenerationConfig, SimpleGenerator
import os
import dataclasses


@dataclasses.dataclass
class MainArgs:
    model_name_or_path: str
    dataset_name: str
    output_dir: str
    tokenizer_name_or_path: str = None
    lora_weights: str = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    prompt_template: int = 0


def build_input(prompt_template, text):
    if prompt_template == 0:
        return f"Simplify the following text. Write only the response, and in Italian.\n {text}"
    elif prompt_template == 1:
        return f"Semplifica il testo seguente.\n{text}"
    else:
        raise ValueError(f"Invalid prompt template {prompt_template}")


def main():
    parser = HfArgumentParser((MainArgs, DefaultGenerationConfig))
    args, gen_args = parser.parse_args_into_dataclasses()

    print(args)
    print(gen_args)

    if args.dataset_name == "admin-It":
        df = pd.read_csv("./data/admin-It.tsv", sep="\t")
        comp_df = df[df["label"] == "comp"]
        texts = comp_df["text"].tolist()
        simp_df = df[df["label"] == "simp"]
        simp_texts = simp_df["text"].tolist()
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}")

    print("Sample of complex texts", texts[:2])
    print("Sample of simpl texts", simp_texts[:2])

    pipe = SimpleGenerator(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        lora_weights=args.lora_weights,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    print("Building inputs with template")
    inputs = [build_input(args.prompt_template, text) for text in texts]
    print("Sample of inputs", inputs[:2])

    generated_texts = pipe(
        inputs,
        **dataclasses.asdict(gen_args),
        return_full_text=False,
        log_batch_sample=5,
    )

    print(generated_texts[:2])

    print(f"Saving generated texts to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(
        {
            "comp_text": texts,
            "simp_text_gold": simp_texts,
            "simp_text_generated": generated_texts,
        }
    ).to_csv(f"{args.output_dir}/out.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
