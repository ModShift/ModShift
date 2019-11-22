import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    print("Downloading CREMI embeddings ...")
    download_file_from_google_drive("1eOPfoKXmDPnxt_hibRjCMN4K8c4jhYWO", "./data/CREMI/data/CREMI.h5")
    print("... done.")
    print("Downloading ISBI embeddings ...")
    download_file_from_google_drive("1E_OqBdOqEIfrK19H4gOxN2qkGYNbAknR", "./data/ISBI/data/ISBI.h5")
    download_file_from_google_drive("1r5n8ReXsJZXk0xrNsPJZ01SJUCzFRV9E", "./data/ISBI/data/ISBI_embeddings_PCA_8.npy")
    print("... done.")
    print("Downloads done.")
