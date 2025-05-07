def load_data_local(finename: str, data_dir: str) -> Tuple[List[Document], str]:
    file_path = os.path.join(data_dir, finename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents from {file_path}")
    return data, finename.rsplit('.',1)[0].replace('_', '')

def seed_milvus(url: str, collection_name: str, json_file: str, data_dir:str)->Milvus:
    embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002")
    local_data, doc_name = load_data_local(finename=json_file, data_dir=data_dir)
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in local_data]
    print ('documents', documents)
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        url=url,
        drop_old=True,
    )
    vectorstore.add_documents(documents, ids=uuids)
    print('vectorstore', vectorstore)
    return vectorstore

def seed_milvus_live(url: str, collection_name: str, json_file: str, data_dir:str)->Milvus:
    embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002")
    documents = crawler(url)
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in local_data]
    print ('documents', documents)
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        url=url,
        drop_old=True,
    )
    vectorstore.add_documents(documents, ids=uuids)
    print('vectorstore', vectorstore)
    return vectorstore

def connect_to_milvus(url: str, collection_name: str) -> Milvus:
    embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002")
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        url=url,
    )

def main():
    seed_milvus('https://localhost:19530', 'data_test', 'stack.json', 'data')
    seed_milvus_live(url, 'https://localhost:19530', 'data_test_live', 'data')

if __name__ == "__main__":
    main()