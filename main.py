from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import pipeline, set_seed


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


if __name__ == '__main__':
    dataset = load_dataset("cnn_dailymail", version="3.0.0")
    print(dataset)

    sample = dataset["train"][23]

    # Number of words in the article
    # Number of input tokens for the transformer
    # Question: Would the transformer have enough context?
    sample_text = dataset["train"][23]["article"][:2000]
    print(f"\nsample_text: {sample_text}\n")

    summaries = {"baseline": three_sentence_summary(sample_text)}

    set_seed(42)

    pipe = pipeline("summarization", model="t5-small")
    pipe_out = pipe(sample_text)
    summaries["t5-mini"] = "".join(sent_tokenize(pipe_out[0]["summary_text"])).replace(" .", ".\n")
    print("============ t5-mini ============")
    print(summaries["t5-mini"])

    pipe = pipeline("summarization", model="t5-large")
    pipe_out = pipe(sample_text)
    summaries["t5-large"] = "".join(sent_tokenize(pipe_out[0]["summary_text"])).replace(" .", ".\n")
    print("============ t5-large ============")
    print(summaries["t5-large"])

    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    pipe_out = pipe(sample_text)
    summaries["bart"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))
    print("============ bart ============")
    print(summaries["bart"])

    pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    pipe_out = pipe(sample_text)
    summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n").replace(" .", ".\n")
    print("============ pegasus ============")
    print(summaries["pegasus"])
