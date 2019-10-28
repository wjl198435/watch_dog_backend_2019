import requests
import base64
url = "http://localhost:8088"
image = b"09942532"
detector_name = "default"
id = 10
kwargs = {"timeout": 100}
dconfig = 12

def encode_image(image):
    """base64 encode an image stream."""
    return base64.b64encode(image).decode('ascii')

response = requests.post(
    url + "/detect", json={"data": encode_image(image), "detector_name": detector_name, "id": id}, **kwargs)


print(response)