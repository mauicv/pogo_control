from src.channel import Channel

HOST = "192.168.0.27"
POST = 8000

def handle_message(message):
    print(message)

if __name__ == "__main__":
    channel = Channel(host=HOST, port=POST)
    channel.serve(handle_message)
    # servo_interface = ServoInterface(channels=[15, 14])
    # channel.serve(servo_interface.update_angle)