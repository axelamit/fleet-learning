import ray

@ray.remote
class EdgeHandler():
    def __init__(self, node_capacity: int):
        self.lock = ray.Lock()
        self.node_capacity : int = node_capacity
        self.nodes_running : dict = {
            "agx4.nodes.edgelab.network" : 0,
            "agx6.nodes.edgelab.network" : 0,
            "agx9.nodes.edgelab.network" : 0,
            "agx10.nodes.edgelab.network" : 0,
            "orin1.nodes.edgelab.network" : 0,
            "orin2.nodes.edgelab.network" : 0
        }

    @ray.method(num_returns=1)
    def get_available_node(self):
        with self.lock:
            node, running = min(self.nodes_running.items(), key=lambda x: x[1])
            if running < self.node_capacity:
                self.nodes_running[node] = self.nodes_running[node] + 1
                return node
        return self.get_available_node()
    
    def job_done(self, node : str):
        self.nodes_running[node] = self.nodes_running[node] - 1