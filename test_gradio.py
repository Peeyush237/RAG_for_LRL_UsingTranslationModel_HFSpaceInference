import sys
try:
    from gradio_client import Client
except ImportError:
    import os
    os.system(f"{sys.executable} -m pip install gradio_client")
    from gradio_client import Client

print("Testing client...")
try:
    client = Client("Peeyush237/linguabridge-translate")
    res = client.predict("ନମସ୍କାର", api_name="/odia_to_english")
    print("SUCCESS:", res)
except Exception as e:
    print("ERROR:", e)
