import os

from FlagEmbedding import FlagModel


class Embedder:
    """
    BGE Embedding模块
    """

    def __init__(
            self,
            model_name="BAAI/bge-large-en-v1.5",
            use_fp16=True
    ):

        print("Loading Embedding Model...")

        device = os.getenv("RULE_RAG_EMBEDDING_DEVICE", "").strip()
        model_kwargs = {
            "use_fp16": use_fp16 if device != "cpu" else False,
            "batch_size": 1 if device == "cpu" else 256,
        }
        if device:
            model_kwargs["devices"] = device

        self.model = FlagModel(model_name, **model_kwargs)

        print("Embedding Model Loaded.")

    def encode(self, text):
        """
        输入：
            str

        输出：
            numpy.ndarray (1024,)
        """

        embedding = self.model.encode([text], batch_size=1)[0]

        return embedding

    def encode_batch(self, texts):
        """
        输入：
            List[str]

        输出：
            ndarray
        """

        embeddings = self.model.encode(texts)

        return embeddings
