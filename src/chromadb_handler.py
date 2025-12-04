import chromadb

class chromaDBWrapper:
    def __init__(self):
        self.collection = None
    # Collection anlegen (wie eine Tabelle)
    def create_init_Client(self):
        self.client = chromadb.PersistentClient()
    def create_collection(self, collection_name_user):
        self.collection = self.client.create_collection(name=collection_name_user)

    def get_perClient(self):
        return chromadb.PersistentClient("chroma")

    def insert_data(self, data):
        if not data:
            print("data is None")

        self.collection.add(
            data["ids"],
            data["documents"]
            )
        """
        Example_input
        collection.add(
            ids=["1", "2"],
            documents=["Das ist ein Testdokument.", "KI verändert die Welt."],
            )
        """
        

    def query_data(self, queries, results_per_q=4): 
        return self.collection.query(query_texts=queries, n_results=results_per_q)
