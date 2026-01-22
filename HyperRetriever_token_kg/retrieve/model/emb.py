from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Load the model and move it to the GPU
_tokenizer = None
_model = None
_device = None

def get_embedding(sentences, batch_size=32):
    global _tokenizer, _model, _device

    if _model is None:
        # Specify the device to run on, use GPU if available, otherwise use CPU
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and move it to the GPU
        _tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        _model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        _model.eval()
        _model = _model.to(_device)
    """
    Generates embeddings for a list of sentences in batches.

    Args:
        sentences (list of str): The sentences to embed.
        batch_size (int): The number of sentences to process in each batch.

    Returns:
        torch.Tensor: The embeddings of the sentences.
    """
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Generating embeddings"):
        batch_sentences = sentences[i:i + batch_size]
        encoded_input = _tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')

        # Move the input to the same device as the model
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to(_device)

        with torch.no_grad():
            model_output = _model(**encoded_input)
            embeddings = model_output[0][:, 0]  # Take the [CLS] vector
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalize
            all_embeddings.append(embeddings.cpu()) # Move to CPU immediately after computation

    return torch.cat(all_embeddings, dim=0)