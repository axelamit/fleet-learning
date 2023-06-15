import socket
import time


class VirtualVehicle:
    def __init__(
        self,
        server_ip="localhost",
        server_port=65432,
        agx_ip="localhost",
        agx_port=59999,
    ):
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)

        self.AGX_IP = agx_ip
        self.AGX_PORT = agx_port
        self.AGX_ADDR = (self.AGX_IP, self.AGX_PORT)

        self.MSG_LENGTH = 1024
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_agx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.message_to_server = ["HELLO", "RESULTS", "GOODBYE"]
        self.message_from_server = ["DO_TASK", "SESSION DONE"]
        self.message_to_agx = ["HELLO", "TRAIN"]
        self.message_from_agx = ["TASK_SCHEDULED", "RESULTS"]

    def run_client(self):
        self.socket_server.connect(self.SERVER_ADDR)
        self.socket_server.send(self.message_to_server[0].encode("utf-8"))

        while True:

            data = self.socket_server.recv(self.MSG_LENGTH)

            if str(data, "utf-8") == self.message_from_server[0]:
                self._client_do_task(str(data, "utf-8"))

            elif str(data, "utf-8") == self.message_from_server[1]:
                print("[CLIENT] Received msg from server: SESSION DONE")
                print("[CLIENT] Terminating session...")
                break

        self.socket_server.close()
        print("[CLIENT] Server socket closed.")

    def _client_do_task(self, server_message):
        print(f"[CLIENT] Received msg from server: {server_message}")
        self._train_on_AGX()
        self._send_results_to_cloud()

    def _train_on_AGX(self):
        print("[CLIENT] Schedules training task on AGX...")
        time.sleep(3)
        self.socket_agx.connect(self.AGX_ADDR)
        self.socket_agx.send(self.message_to_agx[0].encode("utf-8"))

        while True:
            data = self.socket_agx.recv(self.MSG_LENGTH)
            if str(data, "utf-8") == self.message_from_agx[0]:
                print("[CLIENT] Connection to AGX established.")
                self.socket_agx.send(self.message_to_agx[1].encode("utf-8"))
                print("[CLIENT] Waiting for results...")

            elif str(data, "utf-8") == self.message_from_agx[1]:
                print(f"[CLIENT] Received msg from AGX: {self.message_from_agx[1]}")
                print("[CLIENT] Terminating AGX session...")
                break

        self.socket_agx.close()
        print("[CLIENT] AGX socket closed.")
        print("[CLIENT] Received results from AGX.")

    def _send_results_to_cloud(self):
        print("[CLIENT] Sends results to cloud server...")
        time.sleep(3)
        self.socket_server.send(self.message_to_server[1].encode("utf-8"))
        print("[CLIENT] Done.")


def main():
    client = VirtualVehicle()
    client.run_client()


if __name__ == "__main__":
    main()
