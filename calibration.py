import os
from client.client import Client
from filters.butterworth import ButterworthFilter
import dotenv
dotenv.load_dotenv()


if __name__ == "__main__":
    host = os.getenv("HOST")
    port = int(os.getenv("POST"))

    client = Client(
        host=host,
        port=port
    )
    client.connect()
    butterworth_filter = ButterworthFilter(
        order=4,
        cutoff=2.0,
        fs=50.0,
        num_components=8 # 8 servo motors
    )

    data = client.send_data({})
    print(data)

    client.close()
