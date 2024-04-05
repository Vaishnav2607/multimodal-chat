from llm_chains import load_vectordb, create_embeddings

if __name__ == "__main__":
    vector_db = load_vectordb(create_embeddings())
    output = vector_db.similarity_search("Unit-5-1bee dataset")
    print(output)