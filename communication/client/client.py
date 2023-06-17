import socket
import time


class VirtualVehicle:
    def __init__(
        self,
        server_ip="0.0.0.0",
        server_port=65432,
        agx_ip="0.0.0.0",
        agx_port=59999,
    ):
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)
        self.message_to_server = ["HELLO", "RESULTS", "GOODBYE"]
        self.message_from_server = ["DO_TASK", "SESSION DONE"]

        self.AGX_IP = agx_ip
        self.AGX_PORT = agx_port
        self.AGX_ADDR = (self.AGX_IP, self.AGX_PORT)
        self.message_to_agx = ["HELLO", "TRAIN"]
        self.message_from_agx = ["TASK_SCHEDULED", "RESULTS"]

        self.MSG_LENGTH = 1024
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_agx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run_client(self):
        self.socket_server.connect(self.SERVER_ADDR)
        self.socket_server.send(self.message_to_server[0].encode("utf-8"))

        while True:

            data = self.socket_server.recv(self.MSG_LENGTH)

            if str(data, "utf-8") == self.message_from_server[0]:
                self._client_do_task(str(data, "utf-8"))

            elif str(data, "utf-8") == self.message_from_server[1]:
                print(
                    f"[CLIENT] Received msg from server: {self.message_from_server[1]}"
                )
                print("[CLIENT] Terminating server session...")
                break

        self.socket_server.close()
        print("[CLIENT] Server socket closed.\n")

    def _client_do_task(self, server_message):
        print(f"[CLIENT] Received msg from server: {server_message}")
        self._train_on_AGX()
        self._send_results_to_cloud()

    def _train_on_AGX(self):
        self.socket_agx.connect(self.AGX_ADDR)
        self.socket_agx.send(self.message_to_agx[0].encode("utf-8"))

        while True:
            data = self.socket_agx.recv(self.MSG_LENGTH)
            if str(data, "utf-8") == self.message_from_agx[0]:
                print("[CLIENT] Scheduling training task to AGX...")
                self.socket_agx.send(self.message_to_agx[1].encode("utf-8"))

            elif str(data, "utf-8") == self.message_from_agx[1]:
                print(f"[CLIENT] Received msg from AGX: {self.message_from_agx[1]}")
                print("[CLIENT] Terminating AGX session...")
                break

        self.socket_agx.close()
        print("[CLIENT] AGX socket closed.")

    def _send_results_to_cloud(self):
        print("[CLIENT] Sends results to cloud server...")
        self.socket_server.send(self.message_to_server[1].encode("utf-8"))
        print("[CLIENT] Done.")


def main():
    client = VirtualVehicle()
    client.run_client()


if __name__ == "__main__":
    main()
