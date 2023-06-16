import socket
from _thread import start_new_thread
import threading
import time


class AGX:
    def __init__(self, agx_ip="localhost", agx_port=59999):
        self.AGX_IP = agx_ip
        self.PORT = agx_port
        self.ADDR = (self.AGX_IP, self.PORT)
        self.MSG_LENGTH = 1024
        self.message_to_virtual_vehicle = ["TASK_SCHEDULED", "RESULTS"]
        self.message_from_virtual_vehicle = ["HELLO", "TRAIN"]
        self.BUFFER = []
        self.CLIENT_ID = 0
        self.MAX_BUFFER_SIZE = 100

    def run_scheduler(self):
        self.print_lock = threading.Lock()
        self._open_socket()

        try:
            while True:
                #self._update_que()
                self._process_virtual_vehicle_connections()

        except KeyboardInterrupt as e:
            print("Keyboard interrupt")
            self.s.close()
            print("Socket closed.")

    def _open_socket(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(self.ADDR)
        self.s.listen(self.MAX_BUFFER_SIZE)
        print("[AGX] AGX started")
        print(f"[AGX] Listening on {self.AGX_IP} port {self.PORT}\n")

    def _process_virtual_vehicle_connections(self):
        conn, addr = self.s.accept()
        new_virtual_vehicle = (conn, addr)
        self.CLIENT_ID += 1
        print(f"[AGX] Processing virtual vehicle ID {self.CLIENT_ID}")
        self.process_virtual_vehicle(new_virtual_vehicle)

    def process_virtual_vehicle(self, connected_virtual_vehicle):
        self.agx_task(connected_virtual_vehicle)

    def agx_task(self, connected_virtual_vehicle):
        conn, _ = connected_virtual_vehicle

        while True:
            data = conn.recv(self.MSG_LENGTH)

            if not data:
                print("[AGX] Exit connection with client.\n")
                break

            stop_session = self._parse_virtual_vehicle_message(
                connected_virtual_vehicle, data
            )

            if stop_session:
                print("[AGX] Exit connection with client.\n")
                break

        conn.close()
        self.READY = True

    def _parse_virtual_vehicle_message(self, connected_virtual_vehicle, data):

        virtual_vehicle_message = str(data, "utf-8")
        conn, addr = connected_virtual_vehicle
        stop_session = False

        if virtual_vehicle_message == self.message_from_virtual_vehicle[0]:
            print(f"[AGX] Received msg from virtual vehicle: {virtual_vehicle_message}")
            print("[AGX] Scheduling task on AGX")
            conn.sendall(self.message_to_virtual_vehicle[0].encode("utf-8"))

        elif virtual_vehicle_message == self.message_from_virtual_vehicle[1]:
            print("[AGX] Training started")
            self._train()
            print("[AGX] Training completed")
            print(f"[AGX] Ending session with vv on {addr[0]}.")
            conn.sendall(self.message_to_virtual_vehicle[1].encode("utf-8"))
            stop_session = True

        else:
            print("[AGX] Invalid messaged received from vv.")
            print(
                f"[AGX] Received message: {virtual_vehicle_message}. Ignoring message."
            )
            stop_session = True

        return stop_session

    def _train(self):
        time.sleep(10)


def main():
    scheduler = AGX()
    scheduler.run_scheduler()


if __name__ == "__main__":
    main()
